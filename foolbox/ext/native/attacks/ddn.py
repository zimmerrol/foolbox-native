import numpy as np
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd
import numpy as np
from collections.abc import Callable
from ..criteria import misclassification, TargetedMisclassification, Misclassification


def normalize_l2_norms(x: ep.Tensor) -> ep.Tensor:
    norms = flatten(x).square().sum(axis=-1).sqrt()
    norms = ep.maximum(norms, 1e-12)  # avoid division by zero
    factor = 1 / norms
    factor = atleast_kd(factor, x.ndim)
    return x * factor


class DDNAttack:
    """DDN Attack"""

    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        inputs,
        labels,
        *,
        criterion: Callable = misclassification,
        rescale=False,
        epsilon=2.0,
        init_epsilon=1.0,
        num_steps=10,
        gamma=0.05,
    ):
        assert isinstance(criterion, (Misclassification, TargetedMisclassification))
        targeted = isinstance(criterion, TargetedMisclassification)

        if rescale:
            min_, max_ = self.model.bounds()
            scale = (max_ - min_) * np.sqrt(np.prod(inputs.shape[1:]))
            epsilon = epsilon * scale

        x = ep.astensor(inputs)
        y = ep.astensor(labels)
        assert x.shape[0] == y.shape[0]
        assert y.ndim == 1

        step_size = ep.ones(x, len(x))

        def loss_fn(inputs: ep.Tensor, labels: ep.Tensor) -> ep.Tensor:
            logits = ep.astensor(self.model.forward(inputs.tensor))

            sign = -1.0 if targeted else 1.0
            loss = sign * ep.crossentropy(logits, labels).sum()
            is_adversarial = criterion(x, labels, inputs, logits)

            return loss, is_adversarial

        grad_and_is_adversarial = ep.value_and_grad_fn(x, loss_fn, has_aux=True)

        delta = ep.zeros_like(x)

        epsilon = epsilon * ep.ones(x, len(x))
        worst_norm = flatten(ep.maximum(x, 1 - x)).square().sum(axis=-1).sqrt()

        best_l2 = worst_norm
        best_delta = delta
        adv_found = ep.zeros(x, len(x)).bool()

        for i in range(num_steps):
            x_adv = x + delta

            _, is_adversarial, gradients = grad_and_is_adversarial(x_adv, y)
            gradients = normalize_l2_norms(gradients)

            l2 = ep.norms.l2(flatten(delta), axis=-1)
            is_smaller = l2 < best_l2

            is_both = is_adversarial * is_smaller
            adv_found = ep.logical_or(adv_found, is_adversarial)
            best_l2 = best_l2 * (ep.logical_not(is_both)) + l2 * is_both

            is_both_kd = atleast_kd(is_both, len(x.shape))
            best_delta = best_delta * (ep.logical_not(is_both_kd)) + delta * is_both_kd

            # perform cosine annealing of LR starting from 1.0 to 0.01
            delta = delta + atleast_kd(step_size, x.ndim) * gradients
            step_size = (
                0.01 + (step_size - 0.01) * (1 + np.cos(np.pi * i / num_steps)) / 2
            )

            epsilon = epsilon * (1.0 - (2 * is_adversarial.float32() - 1.0) * gamma)
            epsilon = ep.minimum(epsilon, worst_norm)

            # do step
            delta = delta + atleast_kd(step_size, x.ndim) * gradients

            # clip to valid bounds
            delta = (
                delta
                * atleast_kd(epsilon, x.ndim)
                / delta.square().sum(axis=(1, 2, 3), keepdims=True).sqrt()
            )
            delta = ep.clip(x + delta, *self.model.bounds()) - x

        x_adv = x + delta

        return x_adv.tensor
