"""One identical question through both traditions, for the abstract's own claim.

al-Ayni on Bukhari (Sunni) and al-Mazandarani on al-Kafi (Shia), same question, same
condition, same model. The abstract promises "how that derived method differs across
Sunni and Shia traditions"; this is that, in one slide.

The question asks who inherits prophetic knowledge and what the inheritance authorises.
Both commentaries treat it in their Kitab al-Ilm at comparable length (al-Ayni 16.8K +
4.6K characters, al-Mazandarani 29.1K + 28.9K), and the split is sharp -- the ulama for
the Sunni tradition, the Imams for the Imami one. It is also a question ABOUT
interpretive authority, which is the argument this project is making.

A Python driver rather than a shell script: the question is Arabic, and Windows
PowerShell 5.1 reads a BOM-less .ps1 as ANSI and mangles it into a parse error.

Usage:
  python scripts/run_two_traditions.py            # both arms
  python scripts/run_two_traditions.py --dry-run  # print the commands, call nothing
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hermeneutics.saia import DEFAULT_MODEL, SAIA_BASE_URL, saia_key

QUESTION = ("من هم ورثة علم النبي صلى الله عليه وسلم، "
            "وما الذي تمنحه هذه الوراثة من سلطة في فهم الحديث؟")

ARMS = ["cayni", "mazandarani"]

# SAIA, not poc/.env -- that file points at HUMAIN and returns 403. load_dotenv() does
# not override an env var already set, so setting these here wins.
SAIA = {"VLLM_BASE_URL": SAIA_BASE_URL, "VLLM_MODEL_ID": DEFAULT_MODEL}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--condition", default="agent_method")
    p.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS,
                   help="re-run one arm without paying for the other; SAIA queues for "
                        "20 minutes on a single call, so an arm costs about an hour")
    p.add_argument("--max-tokens", type=int, default=4000)
    args = p.parse_args(argv)

    env = {**os.environ, **SAIA, "PYTHONIOENCODING": "utf-8"}
    key = saia_key()
    if not key and not args.dry_run:
        print("SAIA_API_KEY is not set (User scope). Nothing run.", file=sys.stderr)
        return 1
    env["VLLM_API_KEY"] = key

    failed = []
    for arm in args.arms:
        cmd = [sys.executable, "scripts/run_inquiry.py", "-q", QUESTION,
               "--commentator", arm, "--condition", args.condition,
               "--auto-approve", "--max-tokens", str(args.max_tokens),
               "--run-id", f"twotrad_{arm}"]
        print(f"\n=== {arm} | {datetime.now(timezone.utc).isoformat()} ===", flush=True)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        # Sequential, not parallel: SAIA throttles, run_inquiry cannot resume, and a 429
        # costs the whole arm. Two arms at ~10 min is cheaper than one retry.
        rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        print(f"=== {arm} exit {rc} | {datetime.now(timezone.utc).isoformat()} ===",
              flush=True)
        if rc != 0:
            failed.append(arm)
            # Keep going: one surviving arm is worth having, and the comparison can be
            # rebuilt by re-running only the arm that died.
    if failed:
        print(f"\nFAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
