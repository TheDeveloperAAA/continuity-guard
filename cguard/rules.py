"""Code-only checks (no AI): exact, free and repeatable.

- speaker inventory
- proper-name extraction and spelling-variant clustering
- eye/hair colour mentions attributed to a named character
These also power the 'code-only' mode used when no model key is available.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .parser import Episode

TITLE_WORDS = {
    "LADY", "LORD", "SIR", "THE", "MR", "MRS", "MISS", "MS", "DUKE", "DUCHESS", "DOWAGER", "KING", "QUEEN",
    "PRINCE", "PRINCESS", "COUNT", "COUNTESS", "BARON", "BARONESS", "EARL", "MADAM", "MADAME", "DR", "CAPTAIN",
    "FATHER", "MOTHER", "SISTER", "BROTHER", "YOUNG", "OLD", "ELDER", "NOBLE", "ROYAL",
}
STOP_NAMES = {w.title() for w in TITLE_WORDS} | {
    "Your", "Grace", "Majesty", "Highness", "Lordship", "Ladyship", "God", "Heaven", "Heavens", "Sunday", "Monday",
    "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December", "Episode", "Scene", "Close", "Medium", "Wide",
    "Tight", "Extreme", "Low", "High", "Over", "Insert", "Cut", "Push", "Pull", "Slow", "Fast", "Dutch", "Camera",
    "Action", "Start", "End", "Logline", "Notes", "Continuity", "Sound", "Music", "Okay", "Yes", "Please", "Sorry",
    "Thank", "Thanks", "Good", "Morning", "Night", "Evening", "Christmas", "Manor", "Estate", "Hall", "House", "Castle",
    "Church", "Cathedral", "Chapel", "Village", "Court", "Council", "Crown", "Duchy", "Quick", "Snap", "Rapid",
}
NAME_TOKEN = re.compile(r"\b([A-Z][a-z]{2,}(?:-[A-Z][a-z]+)?)(?:'s)?\b")
COLOURS = r"blue|green|grey|gray|brown|hazel|amber|black|dark|violet|golden"
SHADES = r"dark|light|pale|deep|bright"
EYE_RE = re.compile(rf"\b((?:(?:{SHADES})\s+)?(?:{COLOURS})(?:-(?:{COLOURS}))?)\s+eyes?\b", re.I)
COLOUR_PHRASE = re.compile(rf"\b((?:(?:{SHADES})\s+)?(?:{COLOURS}|silver|white|blond|blonde|auburn|red|raven|chestnut|copper)(?:-(?:{COLOURS}))?)\b", re.I)
HAIR_RE = re.compile(r"\b(silver|grey|gray|white|blond|blonde|golden|dark|black|brown|auburn|red|raven|chestnut|copper)[- ]hair(?:ed)?\b", re.I)
PRONOUNS = re.compile(r"\b(his|her|their)\b", re.I)


def norm_colour(v: str) -> str:
    """'Dark brown' -> 'brown', 'gray' -> 'grey'; a lone shade such as 'dark' is kept."""
    v = v.lower().replace("gray", "grey").replace("blonde", "blond").strip()
    parts = v.split()
    if len(parts) > 1 and re.fullmatch(SHADES, parts[0]):
        v = " ".join(parts[1:])
    return v


def colour_phrase(text: str) -> str | None:
    """The colour words as written (with a shade word if present), e.g. 'dark brown' in 'his dark brown eyes'."""
    m = COLOUR_PHRASE.search(text)
    return m.group(1) if m else None


def strip_titles(label: str) -> str:
    up = label.upper()
    if re.search(r"'S\b", up):
        return up  # a relation such as "ALEXANDER'S MOTHER" names a different person than ALEXANDER
    toks = [t for t in re.split(r"\s+", up) if t and t not in TITLE_WORDS]
    return " ".join(toks)


# ---------------------------------------------------------------- speakers
def speaker_inventory(eps: list[Episode]) -> dict[str, dict]:
    inv: dict[str, dict] = {}
    for e in eps:
        for b in e.blocks:
            if b.kind == "DIALOGUE" and b.speaker:
                d = inv.setdefault(b.speaker, {"lines": 0, "episodes": set(), "first_block": b.id, "sample": b.text[:90]})
                d["lines"] += 1
                d["episodes"].add(e.num)
    for d in inv.values():
        d["episodes"] = sorted(d["episodes"])
    return inv


def speaker_links_code(inv: dict[str, dict]) -> list[tuple[str, str, str]]:
    """Link labels when one is the other plus titles (LADY JULIANA ~ JULIANA)."""
    links = []
    labels = list(inv)
    core = {lb: strip_titles(lb) for lb in labels}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ca, cb = core[a], core[b]
            if not ca or not cb:
                continue
            if ca == cb or (len(ca) > 3 and len(cb) > 3 and (re.search(rf"\b{re.escape(ca)}\b", cb) or re.search(rf"\b{re.escape(cb)}\b", ca))):
                links.append((a, b, "same name after removing titles"))
            elif len(ca) > 3 and len(cb) > 3 and JaroWinkler.normalized_similarity(ca, cb) >= 0.9:
                links.append((a, b, f"labels spelled alike ({JaroWinkler.normalized_similarity(ca, cb):.2f})"))
    return links


# ---------------------------------------------------------------- names
def proper_names(eps: list[Episode]) -> dict[str, list[str]]:
    """Return {name: [unit ids where it occurs]} for capitalised tokens that look like names."""
    units = [u for e in eps for u in e.all_units()]
    corpus_lower = Counter(w for u in units for w in re.findall(r"\b[a-z][a-z]+\b", u.text))
    speaker_tokens = {t.title() for u in units if u.speaker for t in strip_titles(u.speaker).split()}
    occ: dict[str, list[str]] = defaultdict(list)
    mid_sentence: set[str] = set()
    for u in units:
        for m in NAME_TOKEN.finditer(u.text):
            tok = m.group(1)
            if tok in STOP_NAMES or len(tok) < 4:
                continue
            if corpus_lower.get(tok.lower(), 0) > 0 and tok not in speaker_tokens:
                continue  # also used as an ordinary word, so probably not a name
            before = u.text[:m.start()].rstrip()
            if before and before[-1] not in '.!?"\u201c\u2018(:;-\u2014':
                mid_sentence.add(tok)
            if u.id not in occ[tok]:
                occ[tok].append(u.id)
    # a capitalised word that only ever starts a sentence is probably not a name,
    # unless it looks like a misspelling of a name we already trust (that is exactly what we want to catch)
    trusted = {t for t in occ if t in mid_sentence or t in speaker_tokens}
    keep = dict((t, occ[t]) for t in trusted)
    for t, ids in occ.items():
        if t not in trusted and any(JaroWinkler.normalized_similarity(t.lower(), k.lower()) >= 0.85 for k in trusted):
            keep[t] = ids
    return keep


def is_subphrase(a: str, b: str) -> bool:
    """'raichand' and 'raichand group' are the same name written shorter/longer, not a spelling variant."""
    a, b = a.lower().strip(), b.lower().strip()
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return short != long_ and re.search(rf"(?<![\w-]){re.escape(short)}(?![\w-])", long_) is not None


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def name_similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    return max(JaroWinkler.normalized_similarity(a, b), fuzz.ratio(a, b) / 100.0)


def spelling_links(names: list[str], jw_threshold: float = 0.88) -> list[tuple[str, str, float]]:
    """Pairs of names that look like spelling variants of each other."""
    links = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            al, bl = a.lower(), b.lower()
            if al == bl or al.rstrip("s") == bl.rstrip("s") or is_subphrase(al, bl):
                continue
            jw = JaroWinkler.normalized_similarity(al, bl)
            r = fuzz.ratio(al, bl) / 100.0
            if jw >= jw_threshold or (_common_prefix(al, bl) >= 4 and r >= 0.70):
                links.append((a, b, round(max(jw, r), 3)))
    return links


# ---------------------------------------------------------------- traits
def _attribute(text: str, pos: int) -> str:
    """Nearest preceding capitalised name in the same block (code-only heuristic)."""
    before = text[:pos]
    names = [m for m in NAME_TOKEN.finditer(before) if m.group(1) not in STOP_NAMES]
    pron = list(PRONOUNS.finditer(before[-25:]))
    if pron and (not names or names[-1].end() < len(before) - 60):
        return ""  # 'his/her' without a nearby name: code cannot resolve who
    return names[-1].group(1) if names else ""


SCAR_RE = re.compile(r"\b(unscarred|scarred|scars?)\b", re.I)
ITEM_RE = re.compile(r"\b(emerald|ruby|sapphire|gold|golden|silver|oval|square|round|iron|brass|diamond|pearl|jade|copper|bronze)\s+(?:\w+\s+)?(ring|pendant|locket|necklace|crest|brooch|bracelet)\b", re.I)


def trait_mentions_code(eps: list[Episode]) -> list[dict]:
    out = []
    for e in eps:
        for u in e.blocks:
            for rx, trait in ((EYE_RE, "eye_colour"), (HAIR_RE, "hair_colour"), (SCAR_RE, "scar"), (ITEM_RE, "signature_item")):
                for m in rx.finditer(u.text):
                    who = _attribute(u.text, m.start())
                    if not who:
                        continue
                    val = norm_colour(m.group(1)) if trait in ("eye_colour", "hair_colour") else m.group(0).lower()
                    out.append({"character": who, "trait": trait, "value": val, "block": u.id, "phrase": m.group(0)})
    return out


# ---------------------------------------------------------------- extra stat (reported, not a product feature)
def speech_rate_stats(eps: list[Episode]) -> dict:
    rates = []
    for e in eps:
        for b in e.blocks:
            if b.kind == "DIALOGUE" and b.start is not None and b.end and b.end > b.start and b.words >= 3:
                rates.append((b.words / (b.end - b.start), b.id))
    if not rates:
        return {}
    rates.sort(reverse=True)
    vals = sorted(r for r, _ in rates)
    return {
        "dialogue_lines_3plus_words": len(rates),
        "median_words_per_sec": round(vals[len(vals) // 2], 2),
        "lines_over_3_5_wps": sum(r > 3.5 for r, _ in rates),
        "lines_over_4_0_wps": sum(r > 4.0 for r, _ in rates),
        "fastest": [{"block": bid, "words_per_sec": round(r, 1)} for r, bid in rates[:5]],
    }
