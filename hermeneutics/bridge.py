"""A chat model answered by hand instead of by an endpoint.

Why this exists: the frontier-baseline arm has no API key and no funds, but a
Sonnet subagent is reachable from the operator's session. So the pipeline runs
unchanged and its model calls are parked on disk for that operator to answer.

Protocol, one file each way, in STUDY_BRIDGE_DIR:
    req-<seq>.json   {"seq", "system", "user", "model_name"}   written by us
    res-<seq>.json   {"content": "..."}                        written by the operator

NOT a substitute for a real endpoint, and a run driven this way is not a matched
arm: there is no temperature control (spec 3.4 pins temperature=0), the answering
harness wraps its own uncontrolled system prompt around every call, and the model
alias is not a pinned weight. write_bridge_manifest_note() records that beside the
output so no later reader mistakes it for one.
"""
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

POLL_SECONDS = 2.0
# ponytail: an operator answering by hand is slower than any endpoint; 2h is
# "you went to lunch", not "it hung". Lower it if a run is ever unattended.
DEFAULT_TIMEOUT = 7200


class BridgeChatModel(BaseChatModel):
    """Writes each call to a file and blocks until someone writes the answer back."""

    bridge_dir: Path
    model_name: str = "operator-bridge"
    timeout_s: int = DEFAULT_TIMEOUT

    @property
    def _llm_type(self) -> str:
        return "operator-bridge"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # kwargs carries response_format from call_json's .bind(). Dropped on
        # purpose: a human answering cannot be put in JSON mode, and call_json
        # already strips fences and retries once on unparseable output.
        system = "\n".join(m.content for m in messages if m.type == "system")
        user = "\n".join(m.content for m in messages if m.type == "human")

        seq = self._next_seq()
        req = self.bridge_dir / f"req-{seq:04d}.json"
        res = self.bridge_dir / f"res-{seq:04d}.json"
        req.write_text(
            json.dumps(
                {"seq": seq, "model_name": self.model_name,
                 "system": system, "user": user},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

        content = self._await_response(res, seq)
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(
                content=content,
                usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                response_metadata={"finish_reason": "stop", "model_name": self.model_name},
            )
        )])

    def _next_seq(self) -> int:
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        return 1 + max(
            (int(p.stem.split("-")[1]) for p in self.bridge_dir.glob("req-*.json")),
            default=0,
        )

    def _await_response(self, res: Path, seq: int) -> str:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if res.exists():
                # A partially-written file is a real race here: the operator's
                # writer and this poller are different processes. Retry rather
                # than fail the whole run on a half-flushed JSON object.
                try:
                    payload = json.loads(res.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    time.sleep(0.2)
                    continue
                return payload["content"]
            time.sleep(POLL_SECONDS)
        raise TimeoutError(
            f"no res-{seq:04d}.json in {self.bridge_dir} after {self.timeout_s}s"
        )


def bridge_dir_from_env() -> Path | None:
    raw = os.environ.get("STUDY_BRIDGE_DIR")
    return Path(raw) if raw else None
