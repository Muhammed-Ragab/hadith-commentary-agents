"""The bridge blocks on a file and must survive a half-written one."""
import json
import threading
import time
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from hermeneutics.bridge import BridgeChatModel


def answer(bridge: Path, seq: int, content: str, *, torn_first: bool = False):
    """Play the operator: wait for the request, then write the reply."""
    req = bridge / f"req-{seq:04d}.json"
    for _ in range(200):
        if req.exists():
            break
        time.sleep(0.05)
    res = bridge / f"res-{seq:04d}.json"
    if torn_first:
        res.write_text('{"content": "half', encoding="utf-8")
        time.sleep(0.5)
    res.write_text(json.dumps({"content": content}), encoding="utf-8")


def test_round_trip_and_torn_write(tmp_path):
    model = BridgeChatModel(bridge_dir=tmp_path, model_name="sonnet-via-cc", timeout_s=20)
    t = threading.Thread(target=answer, args=(tmp_path, 1, '{"ok": true}'), kwargs={"torn_first": True})
    t.start()
    reply = model.invoke([SystemMessage(content="sys"), HumanMessage(content="usr")])
    t.join()

    assert reply.content == '{"ok": true}'
    assert reply.response_metadata["finish_reason"] == "stop"  # never reads as truncated
    written = json.loads((tmp_path / "req-0001.json").read_text(encoding="utf-8"))
    assert written["system"] == "sys" and written["user"] == "usr"


def test_seq_advances_and_bind_is_tolerated(tmp_path):
    """call_json binds response_format; the bridge must ignore it, not choke."""
    model = BridgeChatModel(bridge_dir=tmp_path, model_name="sonnet-via-cc", timeout_s=20)
    bound = model.bind(response_format={"type": "json_object"})
    for seq in (1, 2):
        t = threading.Thread(target=answer, args=(tmp_path, seq, f"reply {seq}"))
        t.start()
        assert bound.invoke([HumanMessage(content=f"q{seq}")]).content == f"reply {seq}"
        t.join()


def test_timeout_names_the_missing_file(tmp_path):
    model = BridgeChatModel(bridge_dir=tmp_path, model_name="x", timeout_s=1)
    with pytest.raises(TimeoutError, match="res-0001.json"):
        model.invoke([HumanMessage(content="q")])
