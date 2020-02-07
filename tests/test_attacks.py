from typing import List, Tuple, Type
import pytest
import eagerpy as ep

import foolbox.ext.native as fbn

import re

L2 = fbn.types.L2
Linf = fbn.types.Linf

attacks: List[Tuple[fbn.Attack, bool]] = [
    (fbn.attacks.DDNAttack(), True),
    (fbn.attacks.DDNAttack(rescale=True), True),
    (fbn.attacks.InversionAttack(), False),
    (fbn.attacks.L2ContrastReductionAttack(L2(100.0)), False),
    (fbn.attacks.BinarySearchContrastReductionAttack(binary_search_steps=15), False),
    (fbn.attacks.LinearSearchContrastReductionAttack(steps=20), False),
    (fbn.attacks.L2CarliniWagnerAttack(binary_search_steps=3, steps=20), True),
    (fbn.attacks.EADAttack(binary_search_steps=3, steps=20), True),
    (fbn.attacks.EADAttack(binary_search_steps=3, steps=20, decision_rule="L1"), True),
    (fbn.attacks.NewtonFoolAttack(steps=20), True),
    (fbn.attacks.L2ContrastReductionAttack(L2(100.0)).repeat(3), False),
    (fbn.attacks.VirtualAdversarialAttack(iterations=50, xi=1, epsilon=10), True),
    (fbn.attacks.L2BasicIterativeAttack(L2(100.0), stepsize=5.0, steps=10), True),
    (fbn.attacks.LinfBasicIterativeAttack(Linf(1.0), stepsize=5.0, steps=10), True),
    (
        fbn.attacks.ProjectedGradientDescentAttack(Linf(1.0), stepsize=5.0, steps=10),
        True,
    ),
    (fbn.attacks.L2FastGradientAttack(L2(100.0)), True),
    (fbn.attacks.LinfFastGradientAttack(Linf(100.0)), True),
]


@pytest.mark.parametrize("attack_and_grad", attacks)
def test_untargeted_attacks(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor],
    attack_and_grad: Tuple[fbn.Attack, bool],
) -> None:

    attack, attack_uses_grad = attack_and_grad
    fmodel, x, y = fmodel_and_data

    if isinstance(x, ep.NumPyTensor) and attack_uses_grad:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    advs = attack(fmodel, x, y)
    assert fbn.accuracy(fmodel, advs, y) < fbn.accuracy(fmodel, x, y)


targeted_attacks: List[Tuple[fbn.Attack, bool]] = [
    (fbn.attacks.L2CarliniWagnerAttack(binary_search_steps=3, steps=20), True),
    (fbn.attacks.DDNAttack(), True),
    (fbn.attacks.EADAttack(binary_search_steps=3, steps=20), True),
]


@pytest.mark.parametrize("attack_and_grad", targeted_attacks)
def test_targeted_attacks(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor],
    attack_and_grad: Tuple[fbn.Attack, bool],
) -> None:

    attack, attack_uses_grad = attack_and_grad
    fmodel, x, y = fmodel_and_data

    if isinstance(x, ep.NumPyTensor) and attack_uses_grad:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    num_classes = fmodel(x).shape[-1]
    target_classes = (y + 1) % num_classes
    criterion = fbn.TargetedMisclassification(target_classes)
    advs = attack(fmodel, x, criterion)

    adv_before_attack = criterion(x, fmodel(x))
    adv_after_attack = criterion(advs, fmodel(advs))
    assert adv_after_attack.sum().item() > adv_before_attack.sum().item()


attacks_init_raises_exception: List[Tuple[Type[fbn.Attack], dict, Type, str, bool]] = [
    (
        fbn.attacks.EADAttack,
        dict(binary_search_steps=3, steps=20, decision_rule="L2"),
        ValueError,
        "invalid decision rule",
        True,
    ),
]


@pytest.mark.parametrize(
    "attack_exception_text_and_grad", attacks_init_raises_exception
)
def test_attacks_init_raises_exception(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor],
    attack_exception_text_and_grad: Tuple[fbn.Attack, Type, str, bool],
) -> None:

    (
        attack_type,
        attack_arguments,
        target_exception_type,
        target_exception_text,
        attack_uses_grad,
    ) = attack_exception_text_and_grad

    with pytest.raises(target_exception_type) as excinfo:
        attack_type(**attack_arguments)

    exception_text = str(excinfo.value)
    exception_text_found = re.search(target_exception_text, exception_text) is not None
    assert exception_text_found


targeted_attacks_raises_exception: List[Tuple[Type[fbn.Attack], bool]] = [
    (fbn.attacks.EADAttack(), True),
]


@pytest.mark.parametrize(
    "attack_exception_text_and_grad", targeted_attacks_raises_exception
)
def test_targeted_attacks_call_raises_exception(
    fmodel_and_data: Tuple[fbn.Model, ep.Tensor, ep.Tensor],
    attack_exception_text_and_grad: Tuple[fbn.Attack, bool],
) -> None:

    attack, attack_uses_grad = attack_exception_text_and_grad
    fmodel, x, y = fmodel_and_data

    if isinstance(x, ep.NumPyTensor) and attack_uses_grad:
        pytest.skip()

    x = (x - fmodel.bounds.lower) / (fmodel.bounds.upper - fmodel.bounds.lower)
    fmodel = fmodel.transform_bounds((0, 1))

    num_classes = fmodel(x).shape[-1]
    target_classes = (y + 1) % num_classes
    invalid_target_classes = ep.concatenate((target_classes, target_classes), 0)
    invalid_targeted_criterion = fbn.TargetedMisclassification(invalid_target_classes)

    class DummyCriterion(fbn.Criterion):
        def __repr__(self) -> str:
            return ""

        def __call__(
            self, perturbed: fbn.criteria.T, outputs: fbn.criteria.T
        ) -> fbn.criteria.T:
            return ep.zeros(perturbed, len(perturbed)).double()

    invalid_criterion = DummyCriterion()

    # check if targeted attack criterion with invalid number of classes is rejected
    with pytest.raises(ValueError):
        attack(fmodel, x, invalid_targeted_criterion)

    # check if only the two valid criteria are accepted
    with pytest.raises(ValueError):
        attack(fmodel, x, invalid_criterion)
