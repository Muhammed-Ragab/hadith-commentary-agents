"""OpenRouter access: endpoint and key. Same shape as saia.py so the UI and the
CLI read either backend's key the same way.

OpenRouter is the second backend (spec: Part 1) -- it lets a frontier model with no
SAIA deployment run a `no_sources` arm, and it lets a Qwen run pin a specific upstream
serving stack (deepinfra, bf16) instead of whichever one OpenRouter happens to route to.
"""
import os

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
DEFAULT_PROVIDER_ROUTE = "deepinfra"
DEFAULT_QUANTIZATION = "bf16"


def openrouter_key() -> str:
    """From the User-scope variable, read fresh.

    Same fallback as saia_key(): a shell started before the key was set holds a
    stale copy, so fall back to the registry rather than failing on a technicality.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, "OPENROUTER_API_KEY")[0]
    except (ImportError, OSError, FileNotFoundError):
        return ""


#: Reserved --provider-route value: cheapest available upstream, no pin, no
#: quantization filter. Meant for the frontier model (openai/gpt-5.6-luna) in
#: condition no_sources.
PRICE_SORT_ROUTE = "price"


def provider_body(route: str | None, quantization: str = DEFAULT_QUANTIZATION) -> dict:
    """The "provider" block OpenRouter reads to pin (or not pin) an upstream stack.

    allow_fallbacks=False is the point of pinning: a silent fallback to a different
    serving stack would mix stacks within one run's outputs. A frontier model run
    with no route requested (route=None) gets no pin at all -- only the privacy flag.

    route="price" (PRICE_SORT_ROUTE) is the third case: also no pin, but instead of
    leaving OpenRouter's own default routing in charge, it tells OpenRouter to sort
    candidate upstreams by price and take the cheapest -- quantization means nothing
    here either, since there is no pinned stack to filter it against.
    """
    if route == PRICE_SORT_ROUTE:
        return {"sort": "price", "data_collection": "deny"}
    if route is None:
        return {"data_collection": "deny"}
    return {
        "order": [route],
        "allow_fallbacks": False,
        "quantizations": [quantization],
        "data_collection": "deny",
    }
