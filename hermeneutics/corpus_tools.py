"""Read-only access to local research data."""
import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path


SOURCES = {
    "commentary": "data/babs/*_babs.json",
    "hadith": "data/*_raw.json",
    # Addressable one at a time: the matn stage reads the corpus its commentator
    # actually commentates, and "hadith" is Bukhari and al-Kafi at once. The key is
    # muslim_matn, not muslim, because "muslim" already means the cluster file below
    # and silently repointing it would change what every existing search returns.
    "bukhari": "data/bukhari_raw.json",
    "kafi": "data/kafi_raw.json",
    "muslim_matn": "data/openiti/muslim_raw.json",
    "quran": "data/quran/*.json",
    "lexica": "data/lexica/*.json",
    "rijal": "data/rijal/*.json",
    "muslim": "data/openiti/muslim_clusters.json",
    "metrics": "outputs/networks/*_metrics.json",
}
_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
_ALIF = re.compile(r"[\u0622\u0623\u0625]")
_WORDS = re.compile(r"[A-Za-z0-9_\u0621-\u064a]+")
_STOPWORDS = {"قارن", "بين", "رؤية", "العلاقة", "مع", "ذكر", "المصادر", "ما", "هل", "عن", "في", "من", "على"}
# The corpora carry no author field, so the commentator lives only in the filename.
_FILE_NAMES = {
    "cayni": "العيني عمدة القاري",
    "nawawi": "النووي شرح صحيح مسلم",
    "mazandarani": "المازندراني شرح الكافي",
    "bukhari": "البخاري",
    "kafi": "الكافي",
}
CHUNK, STEP = 1500, 1200


def _normalise(text: str) -> str:
    # "~~" is an OpenITI line-wrap artifact sitting MID-WORD in the raw hadith corpora
    # (bukhari 33,160 occurrences, kafi 68,067). Without stripping it, verify_quote
    # fails a model that quotes "\u0645\u0627 \u0646\u0648\u0649" correctly and passes one that reproduces
    # "\u0645\u0627 ~~\u0646\u0648\u0649" -- the gate would censor honest quotations and reward corpus noise.
    text = _DIACRITICS.sub("", unicodedata.normalize("NFKC", text)).replace("~~", "")
    return _ALIF.sub("\u0627", text).replace("\u0629", "\u0647").replace("\u0649", "\u064a").lower()


def _terms(query: str) -> list[str]:
    terms = []
    for word in _WORDS.findall(_normalise(query)):
        if word in _STOPWORDS or _normalise(word) in _STOPWORDS:
            continue
        # An English question ("is it permissible...") is mostly stopwords the Arabic
        # list never covers, and its short function words ("a", "an", "is", "to") are
        # not search terms -- they are also substrings of Arabic-transliterated
        # filenames such as "mazandarani", which silently narrowed a whole search to
        # one commentator's files (see the whole-token fix in CorpusTools.search).
        if word[:1].isascii() and len(word) < 3:
            continue
        if word.startswith("و") and len(word) > 3:
            word = word[1:]
        if word.startswith("ال") and len(word) > 4:
            word = word[2:]
        terms.append(word)
    return terms


def _records(value):
    if isinstance(value, list):
        return [record for item in value for record in _records(item)]
    if isinstance(value, dict):
        nested = [item for item in value.values() if isinstance(item, (dict, list))]
        return [record for item in nested for record in _records(item)] or [value]
    return []


# hadith_number, not hadith_num: the raw corpora spell it out, so every hadith cited
# from bukhari/kafi/muslim_matn fell through to the bare filename. "Bukhari 1" is a
# citation a scholar can check; "bukhari_raw" is the unverifiable answer this project
# exists to argue against.
_REFERENCE_KEYS = ("kitab", "title", "bab", "bab_ref", "id", "hadith_num", "hadith_number")


def _reference(record: dict, fallback: str) -> str:
    parts = [str(record[key])[:120] for key in _REFERENCE_KEYS if record.get(key)]
    return " — ".join(parts) or fallback


def _record_text(record: dict) -> str:
    for key in ("text", "matn", "arabic", "content", "definition", "entry"):
        if isinstance(record.get(key), str):
            return record[key]
    return json.dumps(record, ensure_ascii=False)


@lru_cache(maxsize=None)
def _chunks(path_str: str, alias: str) -> tuple:
    """Split every record into overlapping windows, pre-normalised for scoring.

    ponytail: caches the normalised corpus in RAM (~2x the JSON on disk) and never
    reloads it. The PoC data is read-only; restart the server if it changes.
    """
    path = Path(path_str)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    out = []
    for record in _records(data):
        if not isinstance(record, dict):
            continue
        reference = _reference(record, path.stem)
        ref_text = _normalise(reference)
        text = _record_text(record)
        for start in range(0, max(len(text), 1), STEP):
            window = text[start:start + CHUNK]
            if not window.strip():
                continue
            out.append((ref_text, _normalise(window), reference, window))
            if start + CHUNK >= len(text):
                break
    return tuple(out)


@lru_cache(maxsize=None)
def _documents(path_str: str) -> tuple:
    """Whole records, normalised the way the study pipeline normalises evidence.

    Separate from _chunks because verification cannot use the search windows: they
    overlap by only CHUNK - STEP = 300 characters, so a quote longer than that can
    straddle a boundary and appear in no single window. Recorded evidence runs to
    1,012 characters, so that miss is not hypothetical -- and a verifier that says
    "not found" for a real quote is worse than no verifier at all.
    """
    path = Path(path_str)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    fold = _grounding().normalize_arabic
    out = []
    for record in _records(data):
        if not isinstance(record, dict):
            continue
        reference = _reference(record, path.stem)
        out.append((reference, fold(_record_text(record))))
        # The bab/kitab heading, as its own searchable record. _record_text returns the
        # body only, so a model that found a heading through search_sources and quoted
        # it -- "باب ما جاء أن الأعمال بالنية والحسبة ولكل امرىء ما نوى" -- was told its
        # correct quotation does not occur in the corpus. _chunks already indexes the
        # reference for search, so the agent could find what the gate then denied.
        #
        # A separate record rather than reference + text concatenated: a needle spanning
        # the junction would match text that exists nowhere, which is the failure mode
        # this whole gate is built to prevent.
        heading = fold(reference)
        if heading:
            out.append((reference, heading))
    return tuple(out)


@lru_cache(maxsize=1)
def _grounding():
    """The study pipeline's grounding module, imported lazily from the PoC.

    Reused rather than reimplemented on purpose: if the agent verified quotes with
    tools._normalise while the metric used grounding.normalize_arabic, "verified"
    on the page and "grounded" in the results table would silently mean two
    different things, and the whole claim rests on them meaning one.
    """
    poc = Path(__file__).resolve().parents[1]
    if str(poc) not in sys.path:
        sys.path.insert(0, str(poc))
    from hermeneutics import grounding
    return grounding


LEXICA = {"lisan": "data/lexica/lisan_entries.json",
          "nihaya": "data/lexica/nihaya_entries.json"}
# Lisan articles run to 60,320 characters (median 651). Whole ones would swamp the
# model's context for no gain, so send a window around the word and say it is a window.
ARTICLE_CHARS = 1400


@lru_cache(maxsize=None)
def _lexicon(path_str: str) -> tuple:
    """(normalised root, entry, section, text) per article. Entries ARE roots here:
    6,547 triliteral and 2,605 quadriliteral in Lisan, which is why root lookup is
    the natural index and surface-form search is only the fallback."""
    try:
        data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    return tuple((_normalise(x.get("entry", "")), x.get("entry", ""),
                  x.get("section", ""), x.get("text", ""))
                 for x in data if isinstance(x, dict))


@lru_cache(maxsize=4096)
def _roots(word: str) -> tuple:
    """Dotted roots from camel-tools, e.g. بوائقه -> ب.#.ق.

    The builtin database leaves weak radicals as '#', so a root is a PATTERN over the
    lexicon's entries, not a key. Returned dotted; the caller turns '#' into a wildcard.
    Optional: without camel-tools installed, surface-form search still answers.
    """
    try:
        from camel_tools.morphology.analyzer import Analyzer
        from camel_tools.morphology.database import MorphologyDB
    except ImportError:
        return ()
    try:
        analyzer = _analyzer(MorphologyDB, Analyzer)
        return tuple(sorted({a["root"] for a in analyzer.analyze(word)
                             if a.get("root")}))
    except Exception:
        return ()


@lru_cache(maxsize=1)
def _analyzer(db_cls, analyzer_cls):
    return analyzer_cls(db_cls.builtin_db())


def _root_pattern(dotted: str):
    """ب.#.ق -> ^ب.ق$ over normalised entries."""
    radicals = [r for r in dotted.split(".") if r]
    if not radicals:
        return None
    body = "".join("." if r == "#" else re.escape(_normalise(r)) for r in radicals)
    return re.compile(f"^{body}$")


def _window(text: str, needle: str) -> str:
    at = _normalise(text).find(needle) if needle else -1
    if at < 0:
        return text[:ARTICLE_CHARS]
    start = max(0, at - ARTICLE_CHARS // 3)
    return text[start:start + ARTICLE_CHARS]


METRICS = {"bukhari": "outputs/networks/bukhari_metrics.json",
           "kafi": "outputs/networks/kafi_metrics.json",
           # al-Nawawi's isnad stage runs on Muslim, not Bukhari or al-Kafi -- without
           # this key lookup_isnad_network could never be scoped to his own corpus, the
           # same class of gap TRADITION_SCOPE below closes for search and rijal.
           "muslim": "outputs/networks/muslim_metrics.json"}


# One scope per commentator, defined once here and enforced inside CorpusTools rather
# than scattered across call sites (see CorpusTools.__init__). Every path is concrete
# and checked against the files actually on disk in test_corpus_tools_move.py and
# test_tradition_scope.py -- a metrics or rijal file renamed out from under this dict
# fails loudly there rather than silently widening a run's retrieval.
TRADITION_SCOPE = {
    "cayni": [
        "data/babs/cayni_babs.json",
        "data/bukhari_raw.json",
        "data/rijal/sunni_rijal.json",
        "outputs/networks/bukhari_metrics.json",
    ],
    "nawawi": [
        "data/babs/nawawi_babs.json",
        "data/openiti/muslim_raw.json",
        "data/openiti/muslim_clusters.json",
        "data/rijal/sunni_rijal.json",
        "outputs/networks/muslim_metrics.json",
    ],
    "mazandarani": [
        "data/babs/mazandarani_babs.json",
        "data/kafi_raw.json",
        "data/rijal/imami_rijal.json",
        "outputs/networks/kafi_metrics.json",
    ],
}
# Common ground every tradition may search: scripture and the two general dictionaries
# are not any one commentator's property the way his own babs/rijal/metrics files are.
SHARED_SCOPE = ["data/quran/*.json", "data/lexica/*.json"]


def scope_for(commentator: str) -> list[str]:
    """Patterns (concrete files, plus SHARED_SCOPE's globs) this commentator's run may
    search. Raises for an unconfigured commentator rather than silently scoping to
    nothing, which would look like an empty corpus instead of a missing entry."""
    if commentator not in TRADITION_SCOPE:
        raise ValueError(f"No TRADITION_SCOPE configured for commentator: {commentator}")
    return [*TRADITION_SCOPE[commentator], *SHARED_SCOPE]


@lru_cache(maxsize=None)
def _resolve_patterns(poc_root_str: str, patterns: tuple) -> tuple:
    """The concrete files a tuple of patterns resolves to under poc_root, as absolute
    resolved Path objects -- what CorpusTools filters searches against, and (by name)
    what outside_scope checks a verified quotation's provenance against."""
    root = Path(poc_root_str)
    found = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            found.add(path.resolve())
    return tuple(sorted(found))


def resolve_scope(poc_root, commentator: str) -> list[str]:
    """The concrete files this run may search, as paths relative to poc_root -- what
    the manifest and session JSON record so a reader can check what the agent was
    allowed to see (spec item 5)."""
    root = Path(poc_root).resolve()
    patterns = tuple(scope_for(commentator))
    return sorted(p.relative_to(root).as_posix()
                 for p in _resolve_patterns(str(root), patterns))


def outside_scope(found_in: list[str], commentator: str, poc_root) -> bool:
    """True when a verified quotation's occurrences are ALL in files outside this run's
    scope -- a rival commentary (foreign_commentary already catches that case) but also
    a rival hadith collection, which foreign_commentary deliberately ignores because
    quoting the hadith corpus is legitimate. Both are worth flagging here: the corpus
    is real, but not one this commentator's run was scoped to search.

    poc_root may be missing or the commentator unconfigured (e.g. a test double with no
    filesystem backing); either is treated as "cannot tell", not "outside scope".
    """
    if not found_in or not poc_root or commentator not in TRADITION_SCOPE:
        return False
    root = Path(poc_root).resolve()
    patterns = tuple(scope_for(commentator))
    allowed_names = {p.name for p in _resolve_patterns(str(root), patterns)}
    found_names = {Path(src.split(":", 1)[1] if ":" in src else src).name
                  for src in found_in}
    return bool(found_names) and found_names.isdisjoint(allowed_names)


@lru_cache(maxsize=None)
def _network_nodes(path_str: str) -> tuple:
    """(normalised id, id, degree, betweenness, eigenvector, community) per narrator
    node in the §3.5 export. Absent/unparseable file -> empty tuple, same contract
    as _rijal and _lexicon: a missing artifact returns no matches, not an error."""
    try:
        data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    return tuple((_normalise(n.get("id", "")), n.get("id", ""), n.get("degree", 0.0),
                  n.get("betweenness", 0.0), n.get("eigenvector", 0.0), n.get("community", -1))
                 for n in data.get("nodes", []) if isinstance(n, dict))


RIJAL = {"sunni": "data/rijal/sunni_rijal.json",
         "imami": "data/rijal/imami_rijal.json"}
# Connectives carry no identifying weight and appear in nearly every name, so requiring
# them would match everything. Kept in the displayed label, ignored for scoring.
_NAME_CONNECTIVES = {"بن", "ابن", "ابو", "ابي", "اب", "عبد", "ال"}
# Terms the two traditions actually grade with, from each file's grading_vocabulary
# plus the surrounding vocabulary of jarh wa-ta'dil. Reported as PRESENT IN THE ENTRY,
# never as a verdict: "روى عن فلان الثقة" grades someone else.
GRADING_TERMS = ("ثقة", "صدوق", "ضعيف", "متروك", "كذاب", "مجهول", "حسن",
                 "لا بأس به", "منكر الحديث", "مدلس", "وضاع", "متهم")
ENTRY_CHARS = 1200


@lru_cache(maxsize=None)
def _rijal(path_str: str) -> tuple:
    try:
        data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    entries = data.get("entries", []) if isinstance(data, dict) else []
    return tuple((_normalise(e.get("label", "")), e.get("label", ""), e.get("section", ""),
                  e.get("source", ""), e.get("entry_number", ""), e.get("text", ""))
                 for e in entries if isinstance(e, dict))


def _name_tokens(name: str) -> list[str]:
    return [t for t in _WORDS.findall(_normalise(name))
            if t not in _NAME_CONNECTIVES and len(t) > 1]


class CorpusTools:
    """Search the PoC corpus without modifying it.

    scope: an optional list of path patterns (concrete files or globs, as returned by
    scope_for) this instance may read. None (the default) is the historical
    whole-corpus behaviour, used by tests, the study graph's own scans, and
    verify_quote below -- a `source="rival"` argument narrows WHICH files are read,
    but never widens past scope. verify_quote is deliberately excluded from scope
    filtering: it is the substrate _synthesis_gate uses to tell whether a verified
    quotation came from inside or outside the run's scope (corpus_tools.outside_scope),
    which is only answerable if verification itself still sees the whole corpus.
    """

    def __init__(self, poc_root: Path, scope: list[str] | None = None):
        self.poc_root = Path(poc_root).resolve()
        self.scope = tuple(scope) if scope is not None else None

    def _in_scope(self, path: Path) -> bool:
        if self.scope is None:
            return True
        return path.resolve() in _resolve_patterns(str(self.poc_root), self.scope)

    def search(self, query: str, source: str = "all", limit: int = 8, per_file: int = 2) -> list[dict]:
        if source != "all" and source not in SOURCES:
            raise ValueError(f"Unknown source: {source}")
        terms = _terms(query)
        if not terms:
            return []
        selected = SOURCES if source == "all" else {source: SOURCES[source]}
        files = []
        for name, pattern in selected.items():
            for path in sorted(self.poc_root.glob(pattern)):
                if not self._in_scope(path):
                    continue
                alias = next((value for stem, value in _FILE_NAMES.items() if stem in path.stem), "")
                # A term names a commentator only by a WHOLE token of 3+ characters
                # against the alias/filename tokens -- a substring match let "a" or
                # "an" in an English question match inside "mazandarani" and silently
                # narrow the whole search to that one commentator's files.
                whose_tokens = set(_WORDS.findall(_normalise(f"{alias} {path.stem}")))
                named_here = any(len(term) >= 3 and term in whose_tokens for term in terms)
                files.append((name, path, alias, named_here))
        # Naming a commentator is a filter on which books to read, not just another search term.
        named = [entry for entry in files if entry[3]]
        scored = []
        for name, path, alias, _ in (named or files):
            key = f"{name}:{path.name}"
            for ref_text, text, reference, window in _chunks(str(path), alias):
                score = sum(text.count(term) for term in terms)
                score += 5 * sum(ref_text.count(term) for term in terms)
                if score:
                    hit = {"source": key, "reference": reference, "excerpt": window}
                    # The Arabic name, beside the filename rather than instead of it --
                    # found_in and _foreign_commentary key on `source`. Without it the
                    # model sees only "mazandarani_babs.json", transliterates it back
                    # into Arabic itself, and wrote "مظاندراني" into every stage of the
                    # 2026-09-22 two-tradition session.
                    if alias:
                        hit["author"] = alias
                    scored.append((score, key, hit))
        scored.sort(key=lambda item: item[0], reverse=True)
        seen, spread, rest = {}, [], []
        for _, key, result in scored:
            if seen.get(key, 0) < per_file:
                seen[key] = seen.get(key, 0) + 1
                spread.append(result)
            else:
                rest.append(result)
        return (spread + rest)[:limit]

    def verify_quote(self, quote: str, source: str = "all") -> dict:
        """Does this Arabic text really occur in the corpus, and where?

        This is the first claim of the paper made executable: a chatbot asserts a
        quotation and nothing checks it; here the assertion fails unless the text is
        found. Returns every place it occurs so the answer can cite one.
        """
        if source != "all" and source not in SOURCES:
            raise ValueError(f"Unknown source: {source}")
        grounding = self._grounding_module()
        needle = grounding.normalize_arabic(quote or "")
        if len(needle) < grounding.MIN_EVIDENCE_CHARS:
            return {"verified": False, "occurrences": [],
                    "reason": f"quote is shorter than {grounding.MIN_EVIDENCE_CHARS} characters "
                              "after normalisation; a match would be accidental"}

        selected = SOURCES if source == "all" else {source: SOURCES[source]}
        occurrences = []
        for name, pattern in selected.items():
            for path in sorted(self.poc_root.glob(pattern)):
                for reference, text in _documents(str(path)):
                    if needle in text:
                        occurrences.append({"source": f"{name}:{path.name}",
                                            "reference": reference})
                        if len(occurrences) >= 10:
                            break
        if occurrences:
            return {"verified": True, "occurrences": occurrences}
        return {"verified": False, "occurrences": [],
                "reason": "this text does not occur in the local sources. Do not present it "
                          "as a quotation. Search for the passage and quote what is there."}

    def lookup_word(self, word: str, lexicon: str = "all", limit: int = 4) -> dict:
        """What the dictionaries say about a word -- the commentator's own first move.

        Lexical analysis is the most frequent operation in all three commentaries
        (27 Nawawi / 21 cayni / 34 mazandarani in job 15789363), so the agent should do
        it the way they do it: open the lexicon, not recall a meaning.

        Two paths, deliberately. Root lookup is the right index but camel-tools returns
        partial roots for weak radicals; surface-form search catches what that misses,
        and finds the derived form quoted inside an article. Whichever hits, hits.
        """
        if lexicon != "all" and lexicon not in LEXICA:
            raise ValueError(f"Unknown lexicon: {lexicon}")
        surface = _normalise(word or "")
        if not surface:
            return {"word": word, "roots": [], "articles": [], "reason": "empty word"}

        dotted = _roots(word)
        patterns = [(d, p) for d, p in ((d, _root_pattern(d)) for d in dotted) if p]
        selected = LEXICA if lexicon == "all" else {lexicon: LEXICA[lexicon]}

        # Scored, not two concatenated lists. camel-tools leaves weak radicals as '#',
        # so بوائقه -> ب.#.ق matches بثق, بخق AND بوق, and taking them in file order
        # handed the model two confidently wrong articles. The article that also quotes
        # the surface form is the right one -- that is how the book is used by hand too.
        scored = []
        for name, rel in selected.items():
            path = self.poc_root / rel
            if not path.is_file() or not self._in_scope(path):
                continue
            for norm_entry, entry, section, text in _lexicon(str(path)):
                norm_text = _normalise(text)
                uses = norm_text.count(surface)
                exact = any("#" not in d and p.match(norm_entry) for d, p in patterns)
                loose = any(p.match(norm_entry) for _, p in patterns)
                if exact:
                    score, how = 1000 + uses, "root"
                elif loose and uses:
                    score, how = 500 + uses, "root + the word appears in the article"
                elif loose:
                    score, how = 100, "root pattern only -- weak radical unresolved"
                elif uses:
                    score, how = uses, "surface form only"
                else:
                    continue
                scored.append((score, {
                    "lexicon": name, "entry": entry, "section": section,
                    "matched_by": how, "occurrences_in_article": uses,
                    "article": _window(text, surface),
                    "truncated": len(text) > ARTICLE_CHARS,
                }))

        scored.sort(key=lambda item: item[0], reverse=True)
        articles = [hit for _, hit in scored[:limit]]
        out = {"word": word, "roots": [d.replace(".", "") for d in dotted],
               "articles": articles}
        if not articles:
            out["reason"] = ("no article found. Say the lexica do not cover this word "
                             "rather than supplying a meaning from memory.")
        elif all(a["matched_by"] == "surface form only" for a in articles):
            out["caution"] = ("no root match -- these articles merely contain the word, "
                              "and may be about something else. Check before relying on them.")
        return out

    def lookup_narrator(self, name: str, school: str = "all", limit: int = 5) -> dict:
        """Find a narrator in the rijal literature -- Ibn Hajar for Sunni transmitters,
        al-Najashi and al-Tusi for Imami ones.

        "Is so-and-so reliable?" is the question a chatbot answers by inventing a
        grading. This returns the actual entries, or says none was found.

        Ambiguity is reported, not resolved. Classical names repeat: a query matching
        thirty narrators has thirty answers, and saying so is the correct response.
        """
        if school != "all" and school not in RIJAL:
            raise ValueError(f"Unknown school: {school}")
        tokens = _name_tokens(name)
        if not tokens:
            return {"name": name, "matches": [], "total_matches": 0,
                    "reason": "no identifying name parts -- connectives alone cannot select a narrator"}

        selected = RIJAL if school == "all" else {school: RIJAL[school]}
        complete, partial = [], []
        for key, rel in selected.items():
            path = self.poc_root / rel
            if not path.is_file() or not self._in_scope(path):
                continue
            for norm_label, label, section, source, number, text in _rijal(str(path)):
                hits = sum(1 for t in tokens if t in norm_label)
                if not hits:
                    continue
                row = {"school": key, "label": label, "section": section, "source": source,
                       "entry_number": number,
                       "name_parts_matched": f"{hits}/{len(tokens)}",
                       "grading_terms_in_entry": [g for g in GRADING_TERMS if g in text],
                       "entry": text[:ENTRY_CHARS],
                       "truncated": len(text) > ENTRY_CHARS}
                if hits < len(tokens):
                    partial.append((hits, row))
                    continue
                # Arabic names are compound and order carries identity: الحسن بن محمد is
                # a different man from محمد بن الحسن, and both labels contain both parts.
                # An in-order subsequence match separates them.
                cursor, ordered = 0, True
                for t in tokens:
                    at = norm_label.find(t, cursor)
                    if at < 0:
                        ordered = False
                        break
                    cursor = at + len(t)
                # Contiguity beats order alone: محمد بن الحسن occurs verbatim in the right
                # entries, while أحمد بن محمد بن عون أبو الحسن also has both parts in order.
                phrase = _normalise(name).strip() in norm_label
                row["name_order_matches"] = ordered
                row["full_name_appears_verbatim"] = phrase
                complete.append((1000 * phrase + 100 * ordered, row))

        # A fabricated name matched one part of three in twenty entries before this split.
        # Partial matches are never returned alongside complete ones, and when they are all
        # there is, they are labelled as guesses -- otherwise the tool can never say "no".
        if complete:
            complete.sort(key=lambda item: item[0], reverse=True)
            out = {"name": name, "total_matches": len(complete),
                   "matches": [row for _, row in complete[:limit]]}
            if len(complete) > limit:
                out["caution"] = (f"{len(complete)} entries carry this full name; {limit} are "
                                  "shown. Classical names repeat -- do not assume one person.")
            out["note"] = ("grading_terms_in_entry lists jarh wa-ta'dil vocabulary PRESENT in "
                           "the text. It is not a verdict: an entry may grade a different "
                           "narrator it cites. Read the entry before reporting a grading.")
            return out

        if partial:
            partial.sort(key=lambda item: item[0], reverse=True)
            return {"name": name, "total_matches": 0,
                    "matches": [row for _, row in partial[:limit]],
                    "caution": ("NO entry carries this full name. These share only part of it "
                                "and are probably different people. Do not report any of them "
                                "as the narrator asked about.")}

        return {"name": name, "total_matches": 0, "matches": [],
                "reason": ("no entry in the local rijal works. Say so rather than assessing "
                           "this narrator from memory.")}

    def lookup_isnad_network(self, name: str, corpus: str = "all", limit: int = 5) -> dict:
        """Graph metrics for a narrator in the isnad transmission network -- the
        §3.5 export, read here and never recomputed (defect #9: the abstract's
        claim of "mapping transmission patterns" had nothing behind it before this).

        A centrality/community number is a graph property of the sampled network --
        how connected or bridging this narrator's node is -- not a judgement about
        his reliability. Report it as data, the way lookup_narrator reports grading
        vocabulary as present-in-the-entry rather than as a verdict.
        """
        if corpus != "all" and corpus not in METRICS:
            raise ValueError(f"Unknown corpus: {corpus}")
        tokens = _name_tokens(name)
        if not tokens:
            return {"name": name, "matches": [], "total_matches": 0,
                    "reason": "no identifying name parts -- connectives alone cannot select a narrator"}

        selected = METRICS if corpus == "all" else {corpus: METRICS[corpus]}
        complete, partial = [], []
        for key, rel in selected.items():
            path = self.poc_root / rel
            if not path.is_file() or not self._in_scope(path):
                continue
            for norm_id, node_id, degree, betweenness, eigenvector, community in _network_nodes(str(path)):
                hits = sum(1 for t in tokens if t in norm_id)
                if not hits:
                    continue
                row = {"corpus": key, "narrator": node_id,
                       "name_parts_matched": f"{hits}/{len(tokens)}",
                       "degree": degree, "betweenness": betweenness,
                       "eigenvector": eigenvector, "community": community}
                if hits < len(tokens):
                    partial.append((hits, row))
                    continue
                # Same in-order + verbatim identity check as lookup_narrator: name
                # order carries identity, and both orderings of a compound name can
                # appear as substrings of the same node id.
                cursor, ordered = 0, True
                for t in tokens:
                    at = norm_id.find(t, cursor)
                    if at < 0:
                        ordered = False
                        break
                    cursor = at + len(t)
                phrase = _normalise(name).strip() in norm_id
                row["name_order_matches"] = ordered
                row["full_name_appears_verbatim"] = phrase
                complete.append((1000 * phrase + 100 * ordered, row))

        if complete:
            complete.sort(key=lambda item: item[0], reverse=True)
            out = {"name": name, "total_matches": len(complete),
                   "matches": [row for _, row in complete[:limit]]}
            if len(complete) > limit:
                out["caution"] = (f"{len(complete)} nodes carry this full name; {limit} are "
                                  "shown. Classical names repeat -- do not assume one person.")
            out["note"] = ("degree/betweenness/eigenvector/community are graph properties of "
                           "the sampled transmission network. They are NOT a judgement about "
                           "this narrator's reliability -- do not report a centrality score as "
                           "a grading.")
            return out

        if partial:
            # Names only, under a key that is NOT `matches`, and with every metric
            # removed. A caution string alone was not enough: this project has three
            # times had a model copy what was printed rather than read the instruction
            # beside it (the copied no_move of §8.14, the copied confidence 0.0 of
            # §8.13). Handing it rows of real centrality numbers under `matches` and
            # asking it not to use them invites exactly the fabricated narrator
            # authority this tool exists to make impossible.
            partial.sort(key=lambda item: item[0], reverse=True)
            return {"name": name, "total_matches": 0, "matches": [],
                    "near_misses": [row["narrator"] for _, row in partial[:limit]],
                    "caution": ("NO node carries this full name. The near_misses share "
                                "only part of it, are probably different people, and "
                                "carry no metrics here for that reason. Say the narrator "
                                "was not found.")}

        return {"name": name, "total_matches": 0, "matches": [],
                "reason": ("this narrator does not appear in the local isnad network. Say so "
                           "rather than inventing a transmission role or centrality.")}

    @staticmethod
    def _grounding_module():
        return _grounding()

    @staticmethod
    def tool_specs() -> list[dict]:
        return [{"type": "function", "function": {
            "name": "search_sources",
            "description": "Search the local hadith research corpus and return citable excerpts.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
                "source": {"type": "string", "enum": ["all", *SOURCES]},
            }, "required": ["query"]},
        }}, {"type": "function", "function": {
            "name": "verify_quote",
            "description": (
                "Check that Arabic text you are about to quote really occurs in the local "
                "sources, and get its reference. Call this before presenting ANY quotation. "
                "If it returns verified=false, the text is not in the sources: do not quote it."
            ),
            "parameters": {"type": "object", "properties": {
                "quote": {"type": "string", "description": "the exact Arabic text to check"},
                "source": {"type": "string", "enum": ["all", *SOURCES]},
            }, "required": ["quote"]},
        }}, {"type": "function", "function": {
            "name": "lookup_word",
            "description": (
                "Look up an Arabic word in Lisan al-Arab and al-Nihaya. Use this for ANY "
                "claim about what a word means, which is the commentators' own first move. "
                "If it returns no article, say the lexica do not cover the word."
            ),
            "parameters": {"type": "object", "properties": {
                "word": {"type": "string", "description": "the Arabic word, as it appears"},
                "lexicon": {"type": "string", "enum": ["all", *LEXICA]},
            }, "required": ["word"]},
        }}, {"type": "function", "function": {
            "name": "lookup_narrator",
            "description": (
                "Look up a hadith narrator in the rijal literature (Ibn Hajar for Sunni, "
                "al-Najashi and al-Tusi for Imami). Use this for ANY claim about who a "
                "narrator was or how he was graded. If it returns nothing, say the local "
                "rijal works do not cover him -- never assess a narrator from memory."
            ),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "the narrator's name in Arabic"},
                "school": {"type": "string", "enum": ["all", *RIJAL]},
            }, "required": ["name"]},
        }}, {"type": "function", "function": {
            "name": "lookup_isnad_network",
            "description": (
                "Look up a narrator's position in the isnad transmission network: degree, "
                "betweenness, eigenvector centrality and Louvain community, computed from "
                "the sampled corpus. Use this for ANY claim about how central, bridging, or "
                "well-connected a narrator's transmission role is. These are graph "
                "properties, not a reliability grading. If it returns nothing, say the "
                "local network does not include him -- never estimate a centrality."
            ),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "the narrator's name in Arabic"},
                "corpus": {"type": "string", "enum": ["all", *METRICS]},
            }, "required": ["name"]},
        }}]
