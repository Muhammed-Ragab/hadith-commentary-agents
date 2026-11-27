"""Load each bake-off candidate's config under the container venv. LOGIN NODE ONLY.

Run it inside Apptainer, against the same venv the jobs serve from, from the repository root:

  module load apptainer/1.4.3
  P=$(readlink -f "$PROJECT")   # $PROJECT is a home-side symlink apptainer cannot see
  apptainer exec --bind /mnt/vast-kisski:/mnt/vast-kisski --pwd "$(readlink -f "$PWD")" \
      "$P/Colpali/containers/python-3.11-bookworm.sif" \
      bash -lc "source $P/Colpali/.venv-vllm-container/bin/activate \
                && export HF_HOME=$P/hf_home && python poc/slurm/check_model_configs.py"


Costs nothing and needs no GPU: it reads config.json and asks transformers whether it
recognises the architecture. That is the check that would have caught job 15479179 (a
transformers too old for vllm) and it is the one that catches the opposite failure -
transformers 4.57.1 not knowing an architecture the older one did, or vice versa. No
weights are touched, so this says nothing about whether a model fits in H100 memory.

"""
import json
import os
import sys
from pathlib import Path

CANDIDATES = [
    "ALLaM-7B-Instruct-preview",
    "Falcon-H1-Arabic-34B-Instruct",
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.6-35B-A3B",
    "gemma-4-31B-it",
]

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def resolve_hub() -> Path:
    """Find the hub, tolerating $PROJECT's home-side symlink.

    Inside Apptainer, /mnt/vast-kisski/home/<user>/.project/dir.project does not exist -
    binding /mnt/vast-kisski does not carry the nested mountpoints through - so a hub path
    built from $PROJECT makes every model report NOT STAGED while the weights sit happily
    on the resolved projects-side path. Try the obvious candidates and take the first that
    is really there.
    """
    candidates = [
        Path(root) / "hub"
        for root in (os.environ.get("HF_HOME"), f"{os.environ.get('PROJECT', '')}/hf_home")
        if root
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    if candidates:
        print(f"hub not found at {candidates[0]}")
        print("Inside Apptainer, export a resolved HF_HOME before running:")
        print('  export HF_HOME="$(readlink -f $PROJECT)/hf_home"')
    return candidates[0] if candidates else Path("hub")


def resolve(hub: Path, model_id: str) -> str | None:
    """Mirror resolve_model in _vllm_common.sh: bare snapshot dir, else HF cache repo id."""
    bare = hub / model_id.rsplit("/", 1)[-1]
    if bare.is_dir():
        return str(bare)
    if (hub / f"models--{model_id.replace('/', '--')}").is_dir():
        return model_id
    return None


def read_config_json(path: str, hub: Path) -> dict:
    """Read config.json directly. transformers refuses architectures it does not know, but
    the file still says which they are, and vLLM's own registry is the real gate."""
    root = Path(path)
    if not root.is_dir():  # an HF cache repo id: find the snapshot
        cache = hub / f"models--{path.replace('/', '--')}" / "snapshots"
        snapshots = sorted(cache.glob("*")) if cache.is_dir() else []
        if not snapshots:
            return {}
        root = snapshots[-1]
    config = root / "config.json"
    return json.loads(config.read_text(encoding="utf-8")) if config.is_file() else {}


def main() -> int:
    import transformers
    from transformers import AutoConfig
    from vllm.model_executor.models.registry import ModelRegistry

    hub = resolve_hub()
    import vllm

    print(f"transformers {transformers.__version__}   vllm {vllm.__version__}")
    print(f"hub {hub}\n")

    supported = set(ModelRegistry.get_supported_archs())
    failed = 0
    for model_id in CANDIDATES:
        path = resolve(hub, model_id)
        if path is None:
            print(f"NOT STAGED   {model_id}")
            failed += 1
            continue

        cfg_json = read_config_json(path, hub)
        archs = cfg_json.get("architectures", []) or []
        model_type = cfg_json.get("model_type", "?")
        vllm_ok = bool(archs) and all(a in supported for a in archs)

        try:
            AutoConfig.from_pretrained(path, trust_remote_code=False)
            tf_ok = True
        except Exception:  # noqa: BLE001 - detail already printed by the earlier pass
            tf_ok = False

        verdict = "ok         " if (tf_ok and vllm_ok) else "UNSUPPORTED"
        print(f"{verdict}  {model_id}")
        print(f"             model_type={model_type} archs={archs or '[]'}")
        print(f"             transformers={'yes' if tf_ok else 'NO'}  vllm={'yes' if vllm_ok else 'NO'}")
        if not (tf_ok and vllm_ok):
            failed += 1

    print(f"\n{len(CANDIDATES) - failed}/{len(CANDIDATES)} usable")
    if failed:
        print("\nvLLM knows these architectures; anything above missing from this list needs a")
        print("newer vLLM, not just a newer transformers:")
        print("  " + ", ".join(sorted(
            a for a in supported
            if "Qwen" in a or "Gemma" in a or "Falcon" in a or "Llama" in a
        )))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
