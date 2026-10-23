# revelyt/core/meta.py
class ProviderMeta(type):
    """
    Metaclass to enforce that all LLM Providers implement the required interface.
    Catches incomplete provider implementations at import time rather than runtime.
    """
    def __new__(mcs, name, bases, namespace):
        # Don't check the base class itself
        if name != 'BaseProvider':
            required_methods = ['get_llm', 'invoke_structured']
            for method in required_methods:
                if method not in namespace and not any(hasattr(b, method) for b in bases):
                    raise TypeError(f"Provider '{name}' must implement '{method}()'")
        return super().__new__(mcs, name, bases, namespace)