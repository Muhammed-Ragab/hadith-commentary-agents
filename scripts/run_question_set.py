"""The conference question set (2026-09-24): Query-Agent on OpenRouter/qwen, and a
frontier model with no tools, one run per cell.

Python rather than a shell loop: PowerShell 5.1 mangles Arabic argv (see
run_two_traditions.py). Each run is its own run_inquiry.py process with its own run id,
so traces never append across runs. Worked examples, n = 1 per cell -- not a measurement.

    python scripts/run_question_set.py            # everything not already saved
    python scripts/run_question_set.py --only luna
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

POC = Path(__file__).resolve().parent.parent
OUT = POC / "outputs" / "question_set"

QUESTIONS = {
    "q1_ray":     "ما حكم الرأي والقياس في فهم الدين؟",
    "q2_taarud":  "كيف يُرجَّح بين حديثين متعارضين؟",
    "q3_manaa":   "هل تجوز رواية الحديث بالمعنى؟",
    "q4_qadar":   "هل الإنسان مسيّر أم مخيّر؟",
    "q5_taqiyya": "هل يجوز للمسلم أن يخفي دينه أو رأيه إذا خاف على نفسه؟",
    "q6_dua":     "لماذا لا يُستجاب دعائي؟",
    "q7_warith":  "من هم ورثة علم النبي صلى الله عليه وسلم، وما الذي تمنحه هذه الوراثة من سلطة في فهم الحديث؟",
    "q8_sin":     "هل صحيح أن النبي صلى الله عليه وسلم قال «اطلبوا العلم ولو بالصين»؟",
    "q1_ray_en":     "What is the status of personal opinion (ra'y) and analogy (qiyas) in understanding the religion?",
    "q5_taqiyya_en": "Is a Muslim allowed to hide his faith or his opinion if he fears for his life?",
    "q6_dua_en":     "Why are my prayers (du'a) not answered?",
    # The invariance check's colloquial arm, re-run: the first died on an upstream 504.
    "q5_taqiyya_colloq": "لو خايف على نفسي، ينفع أخبي ديني أو رأيي وما أقولوش قدام الناس؟",
    # Level 2: mushkil al-ḥadīth under an outside (European-ethics) objection. The
    # questions must not hint at variant wordings or at ʿĀʾiša's objection.
    "q9_shum": (
        "نحن مسلمون نعيش في سياق أوروبي، ويُدَّعى أن حديث «الشؤم في ثلاث: المرأة والدار والفرس» "
        "يحطّ من قدر المرأة ويتعارض مع مبدأ المساواة بين الرجل والمرأة. اجمع روايات هذا الحديث "
        "وألفاظه المختلفة، وبيّن درجته من حيث الصحة والضعف عند أهل الحديث، ثم اربط بين نتيجة هذا "
        "التوثيق وفهم الحديث. وبناءً على ذلك قدّم عرضاً تأويلياً لا يُهمل ما قاله الشرّاح المتقدمون، "
        "ويأخذ الإشكال المعاصر بجدية."),
    "q10_salat": (
        "نحن مسلمون نعيش في سياق أوروبي، ويُدَّعى أن حديث «يقطع الصلاةَ المرأةُ والحمارُ والكلبُ» "
        "يضع المرأة في منزلة الحيوان ويتعارض مع كرامة الإنسان. اجمع روايات هذا الحديث وألفاظه "
        "المختلفة، وبيّن درجته من حيث الصحة والضعف عند أهل الحديث، ثم اربط بين نتيجة هذا التوثيق "
        "وفهم الحديث. وبناءً على ذلك قدّم عرضاً تأويلياً لا يُهمل ما قاله الشرّاح المتقدمون، ويأخذ "
        "الإشكال المعاصر بجدية."),
    "q11_umirtu": (
        "نحن مسلمون نعيش في سياق أوروبي، ويُدَّعى أن حديث «أُمرت أن أقاتل الناس حتى يشهدوا أن لا إله "
        "إلا الله» يدعو إلى قتال غير المسلمين حتى يُسلموا، ويتعارض مع حرية الدين والعيش المشترك في "
        "مجتمع تعددي. اجمع روايات هذا الحديث وألفاظه المختلفة، وبيّن درجته من حيث الصحة والضعف عند "
        "أهل الحديث، ثم اربط بين نتيجة هذا التوثيق وفهم الحديث. وبناءً على ذلك قدّم عرضاً تأويلياً لا "
        "يُهمل ما قاله الشرّاح المتقدمون، ويأخذ الإشكال المعاصر بجدية."),
    "q10_salat_en": (
        "We are Muslims living in a European context, and it is claimed that the hadith \"A woman, "
        "a donkey and a dog interrupt the prayer\" places women on a level with animals and "
        "contradicts human dignity. Gather the narrations of this hadith and their variant "
        "wordings, state its authenticity grading according to the hadith scholars, and then "
        "connect the result of this authentication to how the hadith should be understood. On "
        "that basis, offer an interpretive proposal that does not neglect the classical "
        "commentators and takes the modern objection seriously."),
}

# (question id, commentator). The three q5 mazandarani cells (fusha / English /
# colloquial) double as the invariance check. The earlier invariance_* runs predate
# tradition scoping and are superseded by these.
AGENT_CELLS = [
    ("q1_ray", "mazandarani"), ("q1_ray", "cayni"),
    ("q2_taarud", "mazandarani"),
    ("q3_manaa", "mazandarani"), ("q3_manaa", "nawawi"),
    ("q4_qadar", "mazandarani"), ("q4_qadar", "nawawi"),
    ("q5_taqiyya", "cayni"), ("q5_taqiyya", "mazandarani"),
    ("q5_taqiyya_en", "mazandarani"),
    ("q6_dua", "mazandarani"), ("q6_dua", "cayni"),
    ("q7_warith", "mazandarani"), ("q7_warith", "cayni"),
    ("q8_sin", "cayni"),
    ("q1_ray_en", "mazandarani"), ("q6_dua_en", "mazandarani"),
    ("q5_taqiyya_colloq", "mazandarani"),
    ("q9_shum", "cayni"), ("q9_shum", "mazandarani"),
    ("q10_salat", "cayni"), ("q10_salat", "nawawi"), ("q10_salat", "mazandarani"),
    ("q10_salat_en", "mazandarani"),
    ("q11_umirtu", "cayni"), ("q11_umirtu", "nawawi"),
]
LUNA_CELLS = ["q1_ray", "q2_taarud", "q3_manaa", "q4_qadar", "q5_taqiyya", "q6_dua",
              "q7_warith", "q8_sin", "q1_ray_en", "q5_taqiyya_en", "q6_dua_en",
              "q9_shum", "q10_salat", "q11_umirtu", "q10_salat_en"]

QWEN = ["--provider", "openrouter", "--model-id", "qwen/qwen3.8-27b",
        "--provider-route", "deepinfra", "--quantization", "bf16"]
LUNA = ["--provider", "openrouter", "--model-id", "openai/gpt-5.6-luna",
        "--provider-route", "price"]

_launch_lock = threading.Lock()


def run(run_id: str, question: str, extra: list[str]) -> tuple[str, int, str]:
    index = OUT / "index.json"
    log_path = OUT / "logs" / f"{run_id}.log"
    cmd = [sys.executable, "-u", "scripts/run_inquiry.py", "-q", question,
           "--auto-approve", "--run-id", run_id, *extra]
    with _launch_lock:  # session filenames carry a per-second stamp
        time.sleep(2)
        log = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=POC, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    code = proc.wait()
    log.close()
    m = re.search(r"Session saved: (.+)", log_path.read_text(encoding="utf-8"))
    session = m.group(1).strip() if m else ""
    with _launch_lock:
        data = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
        data[run_id] = {"question": question, "exit": code, "session": session}
        index.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{run_id}: exit {code} {Path(session).name if session else 'NO SESSION'}", flush=True)
    return run_id, code, session


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("agent", "luna"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    done = {}
    if (OUT / "index.json").exists():
        done = json.loads((OUT / "index.json").read_text(encoding="utf-8"))

    jobs = []
    if args.only in (None, "luna"):
        for q in LUNA_CELLS:
            # no_sources never reads a corpus; the commentator only labels the session.
            jobs.append((f"set_{q}_luna", QUESTIONS[q],
                         [*LUNA, "--condition", "no_sources", "--commentator", "mazandarani"]))
    if args.only in (None, "agent"):
        for q, c in AGENT_CELLS:
            jobs.append((f"set_{q}_{c}", QUESTIONS[q],
                         [*QWEN, "--condition", "agent_method", "--commentator", c]))
    jobs = [j for j in jobs if not (done.get(j[0], {}).get("exit") == 0
                                    and done[j[0]].get("session"))]
    print(f"{len(jobs)} runs to go", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda j: run(*j), jobs))
    failed = [r for r in results if r[1] != 0]
    print(f"done: {len(results) - len(failed)} ok, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
