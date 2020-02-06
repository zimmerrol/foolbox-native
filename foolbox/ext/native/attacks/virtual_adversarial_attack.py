from typing import Union
import eagerpy as ep

from ..models import Model

from ..criteria import Misclassification

from ..devutils import flatten

from .base import FixedEpsilonAttack
from .base import get_criterion
from .base import T


class VirtualAdversarialAttack(FixedEpsilonAttack):
    """Calculate an untargeted adversarial perturbation by performing a
    approximated second order optimization step on the KL divergence between
    the unperturbed predictions and the predictions for the adversarial
    perturbation. This attack was introduced in [1]_.

    References
    ----------
    .. [1] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae,
           Shin Ishii,
           "Distributional Smoothing with Virtual Adversarial Training",
           https://arxiv.org/abs/1507.00677
    """

    def __init__(self, xi: float = 1e-6, iterations: int = 1, epsilon: float = 0.3):
        self.xi = xi
        self.iterations = iterations
        self.epsilon = epsilon

    def __call__(
        self, model: Model, inputs: T, criterion: Union[Misclassification, T]
    ) -> T:
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion

        N = len(x)

        if isinstance(criterion_, Misclassification):
            classes = criterion_.labels
        else:
            raise ValueError("unsupported criterion")

        if classes.shape != (N,):
            raise ValueError(
                f"expected labels to have shape ({N},), got {classes.shape}"
            )

        bounds = model.bounds

        def loss_fun(delta: ep.Tensor, logits: ep.Tensor) -> ep.Tensor:
            assert x.shape[0] == logits.shape[0]
            assert delta.shape == x.shape

            x_hat = x + delta
            logits_hat = model(x_hat)
            loss = ep.kl_div_with_logits(logits, logits_hat).sum()

            return loss

        value_and_grad = ep.value_and_grad_fn(x, loss_fun, has_aux=False)

        clean_logits = model(x)

        # start with random vector as search vector
        d = ep.normal(x, shape=x.shape, mean=0, stddev=1)
        for it in range(self.iterations):
            # normalize proposal to be unit vector
            d = d * self.xi / ep.sqrt((d ** 2).sum(keepdims=True, axis=(1, 2, 3)))

            # use gradient of KL divergence as new search vector
            _, grad = value_and_grad(d, clean_logits)
            d = grad

            # rescale search vector
            d = (bounds[1] - bounds[0]) * d

            if ep.any(ep.norms.l2(flatten(d), axis=-1) < 1e-16):
                raise RuntimeError(
                    "Gradient vanished; this can happen if xi is too small."
                )

        final_delta = (
            self.epsilon / ep.sqrt((d ** 2).sum(keepdims=True, axis=(1, 2, 3))) * d
        )
        x_adv = ep.clip(x + final_delta, *bounds)

        return restore_type(x_adv)
