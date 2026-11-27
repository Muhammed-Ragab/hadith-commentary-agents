"""SAIA model listing. The UI must start even when SAIA is unreachable."""
from hermeneutics.saia import DEFAULT_MODEL, list_models


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_list_models_returns_sorted_unique_ids():
    seen = {}

    def get(url, headers, timeout):
        seen.update(url=url, headers=headers)
        return _Resp({"data": [{"id": "qwen3.8-27b"}, {"id": "llama-3.3-70b-instruct"},
                               {"id": "qwen3.8-27b"}]})

    ids, warning = list_models("https://saia.example/v1", "KEY", client_get=get)

    assert ids == ["llama-3.3-70b-instruct", "qwen3.8-27b"]
    assert warning is None
    assert seen["url"] == "https://saia.example/v1/models"
    assert seen["headers"] == {"Authorization": "Bearer KEY"}


def test_unreachable_saia_falls_back_to_the_default_with_a_warning():
    def get(url, headers, timeout):
        raise ConnectionError("no route to host")

    ids, warning = list_models("https://saia.example/v1", "KEY", client_get=get)

    assert ids == [DEFAULT_MODEL]
    assert "no route to host" in warning


def test_an_empty_list_falls_back_too():
    ids, warning = list_models("https://saia.example/v1", "KEY",
                               client_get=lambda url, headers, timeout: _Resp({"data": []}))
    assert ids == [DEFAULT_MODEL]
    assert warning


def test_the_key_never_appears_in_the_warning():
    def get(url, headers, timeout):
        raise RuntimeError(f"401 for {headers}")   # a careless error message

    _, warning = list_models("https://saia.example/v1", "SECRET-KEY", client_get=get)
    assert "SECRET-KEY" not in warning
