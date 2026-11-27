"""SAIA (GWDG chat-ai) access: endpoint, default model, key, model list.

Moved out of scripts/run_two_traditions.py so the UI and the batch driver read the key
the same way. poc/.env points at HUMAIN and returns 403; SAIA is configured here instead.
"""
import os

import httpx

SAIA_BASE_URL = "https://chat-ai.academiccloud.de/v1"
DEFAULT_MODEL = "qwen3.8-27b"


def saia_key() -> str:
    """From the User-scope variable, read fresh.

    A shell started before the key was set holds a stale copy and gets a 401; that cost
    a run once already. os.environ here is this process's copy, so fall back to the
    registry when it is missing rather than failing on a technicality.
    """
    key = os.environ.get("SAIA_API_KEY")
    if key:
        return key
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, "SAIA_API_KEY")[0]
    except (ImportError, OSError, FileNotFoundError):
        return ""


def list_models(base_url: str = SAIA_BASE_URL, key: str | None = None, *,
                client_get=httpx.get) -> tuple[list[str], str | None]:
    """(model ids, warning). Never raises: a UI that cannot start because SAIA is down
    is worse than one that offers only the default model and says why."""
    key = saia_key() if key is None else key
    try:
        r = client_get(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"},
                       timeout=15)
        r.raise_for_status()
        ids = sorted({m["id"] for m in r.json().get("data", [])})
    except Exception as e:  # noqa: BLE001 -- network, auth, or an unexpected payload shape
        detail = f"{type(e).__name__}: {e}"
        if key:
            detail = detail.replace(key, "***")
        return [DEFAULT_MODEL], f"could not list SAIA models ({detail}); showing the default only"
    if not ids:
        return [DEFAULT_MODEL], "SAIA returned no models; showing the default only"
    return ids, None
