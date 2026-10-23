"""
ProviderFactory — self-registering Strategy Pattern for LLM providers.

Adding a new provider:
  1. Create providers/my_provider.py
  2. Decorate the class with @register("alias1", "alias2", ...)
  3. Done — zero changes needed here or in cli.py
"""

import importlib
from typing import Type, Dict

from revelyt.providers.base import BaseProvider

# alias → provider class
_REGISTRY: Dict[str, Type[BaseProvider]] = {}

# All provider modules to auto-import on first use.
# This is the ONLY place new provider modules are listed.
_PROVIDER_MODULES = [
    "revelyt.providers.google_provider",
    "revelyt.providers.openai_provider",
    "revelyt.providers.anthropic_provider",
    "revelyt.providers.huggingface_provider",
    "revelyt.providers.ollama_provider",
    "revelyt.providers.openrouter_provider",
    "revelyt.providers.llamacpp_provider",
    "revelyt.providers.llamacpp_auditor_provider",
]

_providers_loaded = False


def _ensure_providers_loaded() -> None:
    """
    Lazily imports all provider modules so their @register decorators fire.
    Safe to call multiple times — imports happen only once.
    """
    global _providers_loaded
    if _providers_loaded:
        return
    for module_path in _PROVIDER_MODULES:
        importlib.import_module(module_path)
    _providers_loaded = True


def register(*aliases: str):
    """
    Class decorator that registers a provider under one or more CLI aliases.

    Usage:
        @register("google", "gemini", "gemini-pro")
        class GoogleProvider(BaseProvider):
            ...
    """

    def decorator(cls: Type[BaseProvider]) -> Type[BaseProvider]:
        for alias in aliases:
            _REGISTRY[alias.lower()] = cls
        return cls

    return decorator


class ProviderFactory:
    @staticmethod
    def create(provider_name: str, **kwargs) -> BaseProvider:
        """
        Instantiate a provider by any of its registered aliases.

        Raises ValueError with a helpful message if the alias is unknown.
        """
        _ensure_providers_loaded()

        cls = _REGISTRY.get(provider_name.lower())
        if cls is None:
            available = sorted(_REGISTRY.keys())
            raise ValueError(
                f"Unknown provider '{provider_name}'.\n"
                f"Available aliases: {', '.join(available)}\n"
                f"Run `revelyt providers` for the full list with descriptions."
            )
        return cls(model_name=provider_name, **kwargs)

    @staticmethod
    def list_providers() -> Dict[str, Type[BaseProvider]]:
        """
        Returns the registry after ensuring all providers are loaded.
        Useful for introspection (e.g. the `revelyt providers` CLI command).
        """
        _ensure_providers_loaded()
        return dict(_REGISTRY)
