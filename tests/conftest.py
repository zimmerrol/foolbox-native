from typing import Optional, Callable, Tuple, Dict, Any
import functools
import pytest
import eagerpy as ep

import foolbox
import foolbox.ext.native as fbn

ModelAndData = Tuple[fbn.Model, ep.Tensor, ep.Tensor]

models: Dict[str, Callable[..., ModelAndData]] = {}
models_for_attacks = []


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--backend")
    parser.addoption("--skipslow", action="store_true")


@pytest.fixture(scope="session")
def dummy(request: Any) -> ep.Tensor:
    backend: Optional[str] = request.config.option.backend
    if backend is None:
        pytest.skip()
        assert False
    return ep.utils.get_dummy(backend)


def register(backend: str, *, attack: bool = True) -> Callable[[Callable], Callable]:
    def decorator(f: Callable[[Any], ModelAndData]) -> Callable[[Any], ModelAndData]:
        @functools.wraps(f)
        def model(request: Any) -> ModelAndData:
            if request.config.option.backend != backend:
                pytest.skip()
            return f(request)

        global models
        global models_for_attacks

        models[model.__name__] = model
        if attack:
            models_for_attacks.append(model.__name__)
        return model

    return decorator


def pytorch_simple_model(
    device: Any = None, preprocessing: fbn.types.Preprocessing = None
) -> ModelAndData:
    import torch

    class Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
            x = torch.mean(x, 3)
            x = torch.mean(x, 2)
            return x

    model = Model().eval()
    bounds = (0, 1)
    fmodel = fbn.PyTorchModel(
        model, bounds=bounds, device=device, preprocessing=preprocessing
    )

    x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = fmodel(x).argmax(axis=-1)
    return fmodel, x, y


@register("pytorch")
def pytorch_simple_model_default(request: Any) -> ModelAndData:
    return pytorch_simple_model()


@register("pytorch")
def pytorch_simple_model_default_flip(request: Any) -> ModelAndData:
    return pytorch_simple_model(preprocessing=dict(flip_axis=-3))


@register("pytorch")
def pytorch_simple_model_default_cpu_native_tensor(request: Any) -> ModelAndData:
    import torch

    mean = 0.05 * torch.arange(3).float()
    std = torch.ones(3) * 2
    return pytorch_simple_model("cpu", preprocessing=dict(mean=mean, std=std, axis=-3))


@register("pytorch", attack=False)
def pytorch_simple_model_default_cpu_eagerpy_tensor(request: Any) -> ModelAndData:
    mean = 0.05 * ep.torch.arange(3).float32()
    std = ep.torch.ones(3) * 2
    return pytorch_simple_model("cpu", preprocessing=dict(mean=mean, std=std, axis=-3))


@register("pytorch", attack=False)
def pytorch_simple_model_string(request: Any) -> ModelAndData:
    return pytorch_simple_model("cpu")


@register("pytorch", attack=False)
def pytorch_simple_model_object(request: Any) -> ModelAndData:
    import torch

    return pytorch_simple_model(torch.device("cpu"))


@register("pytorch")
def pytorch_resnet18(request: Any) -> ModelAndData:
    if request.config.option.skipslow:
        pytest.skip()

    import torchvision.models as models

    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = fbn.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    x, y = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(y)
    return fmodel, x, y


def tensorflow_simple_sequential(
    device: Optional[str] = None, preprocessing: fbn.types.Preprocessing = None
) -> ModelAndData:
    import tensorflow as tf

    with tf.device(device):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GlobalAveragePooling2D())
    bounds = (0, 1)
    fmodel = fbn.TensorFlowModel(
        model, bounds=bounds, device=device, preprocessing=preprocessing
    )

    x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = fmodel(x).argmax(axis=-1)
    return fmodel, x, y


@register("tensorflow")
def tensorflow_simple_sequential_cpu(request: Any) -> ModelAndData:
    return tensorflow_simple_sequential("cpu", None)


@register("tensorflow")
def tensorflow_simple_sequential_native_tensors(request: Any) -> ModelAndData:
    import tensorflow as tf

    mean = tf.zeros(1)
    std = tf.ones(1) * 255.0
    return tensorflow_simple_sequential("cpu", dict(mean=mean, std=std))


@register("tensorflow", attack=False)
def tensorflow_simple_sequential_eagerpy_tensors(request: Any) -> ModelAndData:
    mean = ep.tensorflow.zeros(1)
    std = ep.tensorflow.ones(1) * 255.0
    return tensorflow_simple_sequential("cpu", dict(mean=mean, std=std))


@register("tensorflow")
def tensorflow_simple_subclassing(request: Any) -> ModelAndData:
    import tensorflow as tf

    class Model(tf.keras.Model):  # type: ignore
        def __init__(self) -> None:
            super().__init__()
            self.pool = tf.keras.layers.GlobalAveragePooling2D()

        def call(self, x: tf.Tensor) -> tf.Tensor:  # type: ignore
            x = self.pool(x)
            return x

    model = Model()
    bounds = (0, 1)
    fmodel = fbn.TensorFlowModel(model, bounds=bounds)

    x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = fmodel(x).argmax(axis=-1)
    return fmodel, x, y


@register("tensorflow")
def tensorflow_simple_functional(request: Any) -> ModelAndData:
    import tensorflow as tf

    channels = 3
    h = w = 224
    data_format = tf.keras.backend.image_data_format()
    shape = (channels, h, w) if data_format == "channels_first" else (h, w, channels)
    x = x_ = tf.keras.Input(shape=shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs=x_, outputs=x)
    bounds = (0, 1)
    fmodel = fbn.TensorFlowModel(model, bounds=bounds)

    x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = fmodel(x).argmax(axis=-1)
    return fmodel, x, y


@register("tensorflow")
def tensorflow_resnet50(request: Any) -> ModelAndData:
    if request.config.option.skipslow:
        pytest.skip()

    import tensorflow as tf

    model = tf.keras.applications.ResNet50(weights="imagenet")
    preprocessing = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
    fmodel = fbn.TensorFlowModel(model, bounds=(0, 255), preprocessing=preprocessing)

    x, y = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = ep.astensor(y)
    return fmodel, x, y


@register("jax")
def jax_simple_model(request: Any) -> ModelAndData:
    import jax

    def model(x: Any) -> Any:
        return jax.numpy.mean(x, axis=(1, 2))

    bounds = (0, 1)
    fmodel = fbn.JAXModel(model, bounds=bounds)

    x, _ = fbn.samples(
        fmodel, dataset="imagenet", batchsize=16, data_format="channels_last"
    )
    x = ep.astensor(x)
    y = fmodel(x).argmax(axis=-1)
    return fmodel, x, y


def foolbox2_simple_model(channel_axis: int) -> ModelAndData:
    class Foolbox2DummyModel(foolbox.models.base.Model):  # type: ignore
        def __init__(self) -> None:
            super().__init__(
                bounds=(0, 1), channel_axis=channel_axis, preprocessing=(0, 1)
            )

        def forward(self, inputs: Any) -> Any:
            if channel_axis == 1:
                return inputs.mean(axis=(2, 3))
            elif channel_axis == 3:
                return inputs.mean(axis=(1, 2))

        def num_classes(self) -> int:
            return 3

    model = Foolbox2DummyModel()
    fmodel = fbn.Foolbox2Model(model)

    with pytest.warns(UserWarning):
        x, _ = fbn.samples(fmodel, dataset="imagenet", batchsize=16)
    x = ep.astensor(x)
    y = fmodel(x).argmax(axis=-1)
    return fmodel, x, y


@register("numpy")
def foolbox2_simple_model_1(request: Any) -> ModelAndData:
    return foolbox2_simple_model(1)


@register("numpy")
def foolbox2_simple_model_3(request: Any) -> ModelAndData:
    return foolbox2_simple_model(3)


@pytest.fixture(scope="session", params=list(models.keys()))
def fmodel_and_data(request: Any) -> ModelAndData:
    global models
    return models[request.param](request)


@pytest.fixture(scope="session", params=models_for_attacks)
def fmodel_and_data_for_attacks(request: Any) -> ModelAndData:
    global models
    fmodel, x, y = models[request.param](request)
    return fmodel, x[:4], y[:4]
