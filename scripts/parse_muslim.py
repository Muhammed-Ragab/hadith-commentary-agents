"""Parse Ṣaḥīḥ Muslim into clusters of parallel transmission routes."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermeneutics.muslim_schemas import Cluster, Transmission

SOURCE = ROOT / "data" / "openiti" / "muslim_raw.json"
DEST = ROOT / "data" / "openiti" / "muslim_clusters.json"

_VERB = r"(?:حدثنا|حدثني|أخبرنا|أخبرني|أنبأنا)"
_ROUTE = re.compile(rf"(?:ح\s+)?و?{_VERB}")
_QUOTES = re.compile(r"«(.*?)»")
# The narration verb agrees with the transmitters it follows: قالا for two, قالوا for
# three or more, قالت for a woman. Muslim's isnads use all of them (1305/717/1394
# occurrences in the mARkdown). Alternation is longest-first so قال cannot claim the
# prefix of its own inflections, and the boundary stops it matching inside unrelated words.
_SAID = r"(?:قالوا|قالتا|قالا|قالت|قال|يقولون|تقول|يقول)"
_FORMULA = re.compile(rf"(?:أنه\s+)?{_SAID}\b\s*[:،]?")
_PARALLEL = re.compile(r"بهذا الإسناد|بمثل حديث|بمعناه|نحوه|مثله")
_CANONICAL = re.compile(r"\s*\((\d+)\)")


def split_chains(text: str) -> list[str]:
    """Split a hadith into route starts without making taḥwīl its own route."""
    matches = list(_ROUTE.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []
    starts = [0, *[match.start() for match in matches[1:]]]
    return [text[start:end].strip(" ،") for start, end in zip(starts, [*starts[1:], len(text)])]


def split_transmissions(text: str, cluster_ref: str) -> list[Transmission]:
    transmissions = []
    for index, route in enumerate(split_chains(text), 1):
        quotes = _QUOTES.findall(route)
        if quotes:
            matn = max(quotes, key=len).strip()
            isnad = _QUOTES.sub("", route).strip(" ،")
            sibling = False
        else:
            formulas = list(_FORMULA.finditer(route))
            if formulas:
                marker = formulas[-1]
                isnad = route[: marker.start()].strip(" ،")
                matn = route[marker.end() :].strip(" ،")
                sibling = bool(isnad and (not matn or _PARALLEL.search(matn)))
                if sibling:
                    matn = ""
            else:
                isnad, matn = route.strip(" ،"), ""
                sibling = bool(isnad)
        if isnad and (matn or sibling):
            transmissions.append(
                Transmission(
                    ref=f"{cluster_ref}#{index}",
                    isnad=isnad,
                    matn=matn,
                    references_sibling=sibling,
                )
            )
    return transmissions


def parse(source: Path = SOURCE) -> list[Cluster]:
    corpus = json.loads(source.read_text(encoding="utf-8"))
    clusters = []
    # hadith_number restarts in every kitab (6299 hadiths, 522 distinct numbers) and repeats
    # again across babs within a kitab; only (kitab, bab, number) is unique. Without the full
    # path "muslim/2" names 119 clusters and no Claim.source_ref can be checked against the
    # text it cites.
    for kitab_index, chapter in enumerate(corpus["chapters"], 1):
        for bab_index, sub_chapter in enumerate(chapter["sub_chapters"], 1):
            for position, hadith in enumerate(sub_chapter["hadiths"], 1):
                # Position within the bab, not hadith_number: the number repeats both across
                # kitabs and across babs, and 425 (kitab, bab, number) keys still collide.
                cluster_ref = f"muslim/k{kitab_index}/b{bab_index}/{position}"
                isnad = hadith.get("isnad", "")
                canonical = _CANONICAL.match(isnad)
                transmissions = split_transmissions(
                    f"{isnad} {hadith.get('matn', '')}".strip(), cluster_ref
                )
                if transmissions:
                    clusters.append(
                        Cluster(
                            cluster_ref=cluster_ref,
                            muslim_number=canonical.group(1) if canonical else "",
                            kitab=chapter["chapter_name"],
                            bab=sub_chapter["sub_chapter_name"],
                            transmissions=transmissions,
                        )
                    )
    return clusters


def main() -> None:
    clusters = parse()
    DEST.write_text(
        json.dumps([cluster.model_dump() for cluster in clusters], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"{len(clusters)} clusters; {sum(len(cluster.transmissions) for cluster in clusters)} transmissions")
    print(DEST)


if __name__ == "__main__":
    main()
