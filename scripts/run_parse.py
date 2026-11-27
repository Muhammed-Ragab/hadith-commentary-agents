import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.parse_openiti import parse_file

TEXTS = {
    "nawawi": Path("../0676Nawawi.MinhajFiSharhMuslim.JK000137-ara.txt"),
    "cayni":  Path("../0855BadrDinCayni.CumdatQari.Shamela0005756-ara1.txt"),
    # OpenITI 1081MuhammadSalihMazandarani.SharhUsulKafi.Shia001174Vols-ara1
    "mazandarani": Path("../1081Mazandarani.SharhUsulKafi.Shia001174Vols-ara1.txt"),
}

OUT_DIR = Path("data/babs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, path in TEXTS.items():
    print(f"Parsing {name} ({path.name})...")
    babs = parse_file(path)
    out = OUT_DIR / f"{name}_babs.json"
    out.write_text(json.dumps(babs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {len(babs)} bābs written to {out}")
