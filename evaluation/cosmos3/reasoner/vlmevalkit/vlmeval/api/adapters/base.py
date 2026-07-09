_ADAPTER_REGISTRY = {}


def register_adapter(name, factory=None):
    """Register an adapter class/factory under the given name.

    Can be used as a decorator::

        @register_adapter('my_adapter')
        class MyAdapter(ModelAdapter): ...

    Or called directly for variants::

        register_adapter('my_adapter-variant', partial(MyAdapter, flag=True))
    """
    if factory is None:
        def decorator(cls):
            _ADAPTER_REGISTRY[name] = cls
            return cls
        return decorator
    _ADAPTER_REGISTRY[name] = factory
    return factory


def build_adapter(name, **kwargs):
    """Instantiate a registered adapter by name."""
    if name not in _ADAPTER_REGISTRY:
        available = list(_ADAPTER_REGISTRY.keys())
        raise KeyError(f"Adapter '{name}' not found in registry. Available: {available}")
    return _ADAPTER_REGISTRY[name](**kwargs)


def get_adapter_registry():
    """Return a copy of the global adapter registry."""
    return dict(_ADAPTER_REGISTRY)


class ModelAdapter:
    """Base class for model-specific input/output processing hooks.

    Subclasses override only the methods they need. All methods have
    sensible no-op defaults except ``build_prompt``, which raises
    ``NotImplementedError``.

    The ``dump_image_func`` attribute is injected by the wrapper at
    evaluation time via ``set_dump_image``.
    """

    def dump_image(self, line, dataset):
        """Return image path(s) for this sample."""
        return self.dump_image_func(line)

    def override_model_args(self, dataset) -> dict:
        """Return extra generation kwargs for this dataset.

        Recognised keys: ``system_prompt``, ``temperature``, etc.
        """
        return {}

    def use_custom_prompt(self, dataset: str, system_prompt=None) -> bool:
        """Whether to use this adapter's ``build_prompt`` for this dataset."""
        return False

    def build_prompt(self, line, dataset=None):
        """Construct the message list for this sample."""
        raise NotImplementedError

    def process_inputs(self, inputs, dataset=None):
        """Transform the inputs list before HTTP message formatting."""
        return inputs

    def process_payload(self, payload: dict, dataset=None) -> dict:
        """Modify the HTTP payload dict before it is sent."""
        return payload

    def postprocess(self, response: str, dataset=None) -> str:
        """Post-process the raw response string."""
        return response
