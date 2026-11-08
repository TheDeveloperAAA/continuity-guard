"""Turn extracted facts into candidate issues, apply verdicts, build flags, apply human decisions and fixes.

Design rule: the model never writes a quote or a location that reaches the report.
Every flag points to a real block found by code, and its snippet is cut from that block by code.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict

from rapidfuzz.distance import JaroWinkler

from . import rules
from .parser import Episode

PLACE_LIKE = {"family_or_house", "place", "title", "organisation", "estate", "manor", "house", "family"}
SEVERITY = {"DIALOGUE": "High", "CAMERA": "Medium", "ACTION": "Medium", "SOUND": "Medium", "SCENE": "Medium",
            "LOGLINE": "Low", "NOTES": "Low"}
GENERIC_PLACE_WORDS = {"Manor", "Estate", "Hall", "House", "Castle", "Abbey", "Court", "Village", "Wing", "Tower", "Chapel",
                       "Cathedral", "Church", "Gardens", "Garden", "Woods", "Forest", "Street", "Road", "Square", "Palace", "Crest",
                       "Group", "Company", "Corporation", "Corp", "Industries", "Holdings", "Enterprises", "Ltd", "Limited", "Inc"}
DIRECTION_WORDS = {"West", "East", "North", "South", "Great", "Old", "New", "Upper", "Lower", "High", "Royal", "Grand"}
CONTRADICTION_WORDS = re.compile(r"\b(deliberate|intentional|disguise|alias|cover name|pseudonym|nickname|on purpose|undercover)\b", re.I)
ITEM_NOUNS = ("ring", "pendant", "locket", "necklace", "crest", "brooch", "sword", "cane", "bracelet", "earrings")  # ordered: first match wins
ITEM_DESC = {"emerald", "ruby", "sapphire", "gold", "golden", "silver", "oval", "square", "round", "iron", "brass",
             "diamond", "pearl", "jade", "copper", "bronze", "black", "red", "blue", "green"}


class UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def groups(self):
        g = defaultdict(list)
        for i in self.p:
            g[self.find(i)].append(i)
        return list(g.values())


def word_re(s: str) -> re.Pattern:
    return re.compile(rf"(?<![\w-]){re.escape(s)}(?![\w-])")


def snippet(text: str, needle: str, width: int = 55) -> str:
    m = word_re(needle).search(text) if needle else None
    if not m:
        i = text.lower().find(needle.lower()) if needle else -1
        if i < 0:
            return text[:2 * width]
        a, b = i, i + len(needle)
    else:
        a, b = m.start(), m.end()
    s, e = max(0, a - width), min(len(text), b + width)
    return ("..." if s > 0 else "") + text[s:e] + ("..." if e < len(text) else "")


def ep_ranges(nums) -> str:
    nums = sorted(set(nums))
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = n
    if start is not None:
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(out)


def _units(eps):
    return [u for e in eps for u in e.all_units()]


def occurrences(units, name: str):
    rx = word_re(name)
    return [u for u in units if rx.search(u.text)]


# ============================================================================= NAME candidates
def _anchors(refers: set[str], name: str) -> set[str]:
    out = set()
    for r in refers:
        for tok in re.findall(r"\b[A-Z][a-z]{2,}\b", r):
            if tok != name and tok not in rules.STOP_NAMES and tok.upper() not in rules.TITLE_WORDS:
                out.add(tok)
    return out


def name_candidates(eps: list[Episode], reader: dict | None, cfg: dict) -> list[dict]:
    units = _units(eps)
    code_names = rules.proper_names(eps)
    universe: dict[str, list] = {n: [u for u in units if u.id in set(ids)] for n, ids in code_names.items()}
    refers, kinds, source = defaultdict(set), defaultdict(set), defaultdict(set)
    for n in universe:
        source[n].add("code")
    if reader:
        for it in reader["names"]:
            toks = [t for t in re.findall(r"[A-Z][\w-]+", it["name"]) if t.upper() not in rules.TITLE_WORDS and t not in rules.STOP_NAMES]
            if it["kind"] == "person" and len(toks) > 1:
                surfaces = toks  # first name + surname: check each
            else:
                core = [t for t in toks if t not in GENERIC_PLACE_WORDS]
                surfaces = core if len(core) == 1 and core[0] not in DIRECTION_WORDS else [it["name"]]
            for j, s in enumerate(surfaces):
                s = re.sub(r"'s$", "", s.strip())
                if len(s) < 3:
                    continue
                k = it["kind"]
                if len(toks) > 1 and j > 0 and k == "person":
                    k = "family_or_house"  # a surname denotes a family/house
                if s not in universe:
                    universe[s] = occurrences(units, s)
                if it.get("refers_to"):
                    refers[s].add(it["refers_to"])
                if k:
                    kinds[s].add(k)
                source[s].add("ai")
    names = [n for n, occ in universe.items() if occ]
    uf = UF(names)
    why = defaultdict(list)
    for a, b, sim in rules.spelling_links(names, cfg["thresholds"]["spelling_similarity"]):
        uf.union(a, b)
        why[frozenset((a, b))].append(f"spelling similarity {sim}")
    if reader:
        anchors = {n: _anchors(refers[n], n) for n in names}
        place_like = [n for n in names if kinds[n] & PLACE_LIKE]
        for i, a in enumerate(place_like):
            for b in place_like[i + 1:]:
                shared = anchors[a] & anchors[b]
                if shared:
                    uf.union(a, b)
                    why[frozenset((a, b))].append(f"both described in relation to {sorted(shared)[0]}")
    cands = []
    for comp in uf.groups():
        if len(comp) < 2:
            continue
        comp.sort(key=lambda n: (-len(universe[n]), n))
        members = []
        for n in comp:
            occ = universe[n]
            members.append({
                "name": n, "count": len(occ), "episodes": ep_ranges(u.ep for u in occ),
                "spoken_lines": sum(u.kind == "DIALOGUE" for u in occ),
                "kinds": sorted(kinds[n]), "refers_to": sorted(refers[n])[:2],
                "examples": [{"block": u.id, "quote": snippet(u.text, n, 45)} for u in occ[:cfg["judge"]["max_examples_per_member"]]],
            })
        cid = f"N{len(cands) + 1:02d}"
        cands.append({"id": cid, "type": "NAME", "members": [m["name"] for m in members],
                      "source": "+".join(sorted({s for n in comp for s in source[n]})),
                      "links": sorted({w for k, v in why.items() if k <= set(comp) for w in v})[:6],
                      "for_model": {"id": cid, "type": "NAME", "members": members}})
    return cands


# ============================================================================= SPEAKER candidates
def _identity_head(identity: str) -> tuple[set[str], set[str]]:
    head = re.split(r",|\(|;| who | - | – | — ", identity, maxsplit=1)[0]
    names = {t for t in re.findall(r"\b[A-Z][a-z]{2,}\b", head) if t not in rules.STOP_NAMES and t.upper() not in rules.TITLE_WORDS}
    words = {w for w in re.findall(r"[a-z]{4,}", identity.lower())} - {"this", "that", "with", "from", "into", "their", "unnamed", "character"}
    return names, words


def speaker_candidates(eps: list[Episode], reader: dict | None, cfg: dict) -> tuple[list[dict], dict]:
    inv = rules.speaker_inventory(eps)
    labels = list(inv)
    if len(labels) < 2:
        return [], inv
    uf = UF(labels)
    why = defaultdict(list)
    for a, b, r in rules.speaker_links_code(inv):
        uf.union(a, b)
        why[frozenset((a, b))].append(r)
    idents = defaultdict(set)
    if reader:
        for s in reader["speakers"]:
            if s["label"] in inv and s["identity"]:
                idents[s["label"]].add(s["identity"])
        core = {lb: set(rules.strip_titles(lb).title().split()) for lb in labels}
        keys = {lb: [_identity_head(i) for i in idents[lb]] for lb in labels}
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                linked = None
                for na, wa in keys[a]:
                    if na and (na & core[b]) and len(core[b]) <= 2:
                        linked = f"reading pass: {a} is {sorted(na)[0]}"
                    for nb, wb in keys[b]:
                        if nb and (nb & core[a]) and len(core[a]) <= 2:
                            linked = f"reading pass: {b} is {sorted(nb)[0]}"
                        elif na and nb and na & nb:
                            linked = f"reading pass: both are {sorted(na & nb)[0]}"
                        elif not na and not nb and wa and wb and len(wa & wb) / len(wa | wb) >= 0.5:
                            linked = "reading pass: same description"
                if linked:
                    uf.union(a, b)
                    why[frozenset((a, b))].append(linked)
    cands = []
    for comp in uf.groups():
        if len(comp) < 2:
            continue
        comp.sort(key=lambda lb: (-inv[lb]["lines"], lb))
        members = []
        for lb in comp:
            d = inv[lb]
            members.append({"label": lb, "lines": d["lines"], "episodes": ep_ranges(d["episodes"]),
                            "first_block": d["first_block"], "sample_line": d["sample"],
                            "reading_pass_identity": sorted(idents[lb])[:3]})
        cid = f"S{len(cands) + 1:02d}"
        cands.append({"id": cid, "type": "SPEAKER", "members": comp,
                      "source": "ai+code" if reader else "code",
                      "links": sorted({w for k, v in why.items() if k <= set(comp) for w in v})[:6],
                      "for_model": {"id": cid, "type": "SPEAKER", "members": members}})
    return cands, inv


# ============================================================================= TRAIT candidates
EYE_COL = re.compile(r"\b((?:blue|green|grey|gray|brown|hazel|amber|black|dark|violet|golden)(?:-(?:blue|green|grey|gray|brown))?)\b", re.I)
HAIR_COL = re.compile(r"\b(silver|grey|gray|white|blond|blonde|golden|dark|black|brown|auburn|red|raven|chestnut|copper)\b", re.I)


def _norm_value(trait: str, value: str) -> str | None:
    v = value.lower()
    if trait.startswith("eye") or trait.startswith("hair"):
        ph = rules.colour_phrase(v)
        return rules.norm_colour(ph) if ph else None
    if trait.startswith("facial"):
        if re.search(r"clean[- ]shaven|shaven|no beard|beardless", v):
            return "no beard"
        return "beard" if re.search(r"beard|stubble|moustache|mustache", v) else None
    if trait.startswith("scar"):
        if re.search(r"unscarred|no scar|smooth|unblemished", v):
            return "no scar"
        side = re.search(r"\b(left|right)\b", v)
        return f"scar ({side.group(1)})" if side else "scar"
    if trait.startswith("signature"):
        noun = next((n for n in ITEM_NOUNS if n in v), None)
        if not noun:
            return None
        desc = sorted({w for w in re.findall(r"[a-z]+", v) if w in ITEM_DESC})
        return f"{noun}: {'/'.join(desc)}" if desc else None
    return None


def _conflicting(trait: str, values: set[str]) -> bool:
    if len(values) < 2:
        return False
    if trait.startswith("facial"):
        return {"beard", "no beard"} <= values
    if trait.startswith("scar"):
        return ("no scar" in values and len(values) > 1) or ({"scar (left)", "scar (right)"} <= values)
    if trait.startswith("signature"):
        by_noun = defaultdict(set)
        for v in values:
            n, d = v.split(": ", 1)
            by_noun[n].add(d)
        return any(len(ds) > 1 for ds in by_noun.values())
    return True


def trait_candidates(eps: list[Episode], reader: dict | None, identity_map: dict[str, str]) -> list[dict]:
    units = {u.id: u for u in _units(eps)}
    mentions = [dict(t, source="ai") for t in reader["traits"]] if reader else []
    seen = {(m["block"], m["trait"][:3]) for m in mentions}
    for t in rules.trait_mentions_code(eps):  # code findings are always kept; the AI adds to them, never replaces them
        if (t["block"], t["trait"][:3]) not in seen:
            mentions.append({"character": t["character"], "trait": t["trait"], "value": t["phrase"], "source": "code",
                             "deliberate_change": False, "block": t["block"], "ep": units[t["block"]].ep})
    grouped = defaultdict(list)
    for m in mentions:
        who = resolve_person(m["character"], identity_map)
        trait = m["trait"] if m["trait"] != "hair" else "hair_colour"
        trait = "eye_colour" if trait.startswith("eye") else trait
        nv = _norm_value(trait, m["value"])
        if nv is None:
            continue
        grouped[(who, trait)].append({**m, "norm": nv})
    common = {w for e in eps for w in re.findall(r"\b[a-z]+\b", e.raw)}
    cands = []
    for (who, trait), ms in sorted(grouped.items()):
        if who.lower() in common or not who.strip():
            continue  # generic role (several different people), not one character
        vals = {m["norm"] for m in ms if not m.get("deliberate_change")}
        if not _conflicting(trait, vals):
            continue
        by_val = defaultdict(list)
        for m in ms:
            u = units[m["block"]]
            by_val[m["norm"]].append({"block": u.id, "episode": u.ep, "quote": snippet(u.text, m["value"].split()[0], 60),
                                      "as_written": m["value"], "marked_deliberate": m.get("deliberate_change", False)})
        cid = f"T{len(cands) + 1:02d}"
        cands.append({"id": cid, "type": "TRAIT", "character": who, "trait": trait, "values": dict(by_val),
                      "for_model": {"id": cid, "character": who, "trait": trait,
                                    "descriptions": {v: lst[:4] for v, lst in by_val.items()}}})
    return cands


def resolve_person(character: str, identity_map: dict[str, str]) -> str:
    c = character.strip()
    if c.upper() in identity_map:
        return identity_map[c.upper()]
    names, _ = _identity_head(c)
    for n in names:
        if n.upper() in identity_map:
            return identity_map[n.upper()]
    if names:
        return sorted(names)[0]
    return c.title()


# ============================================================================= verdicts -> groups
def _counts(eps, members):
    units = _units(eps)
    return {m: len(occurrences(units, m)) for m in members}


def groups_from_decisions(cands: list[dict], decisions: dict, eps, mode: str, source: str = "ai") -> list[dict]:
    groups = []
    for c in cands:
        d = decisions.get(c["id"]) if decisions else None
        members = c["members"]
        if d is None:
            # no AI answer for this candidate (code-only mode, or the model failed): fall back to the code verdict
            fallback = _code_verdict(c, eps)
            if mode != "code":
                for g in fallback:
                    g["source"] = "code (no AI answer)"
            groups.extend(fallback)
            continue
        seen = set()
        before = len(groups)
        for g in d.get("groups") or []:
            mem = [m for m in (g.get("members") or []) if m in members and m not in seen]
            if not mem:
                continue
            seen.update(mem)
            can = g.get("canonical") if g.get("canonical") in mem else mem[0]
            verdict = str(g.get("verdict", "uncertain")).lower()
            if verdict not in {"inconsistent", "deliberate", "distinct", "uncertain"}:
                verdict = "uncertain"
            try:
                conf = float(g.get("confidence", 0))
            except (TypeError, ValueError):
                conf = 0.0
            groups.append({"members": mem, "canonical": can, "verdict": verdict, "confidence": round(conf, 2),
                           "reason": str(g.get("reason", ""))[:300], "source": source})
        for m in members:
            if m not in seen:
                groups.append({"members": [m], "canonical": m, "verdict": "uncertain", "confidence": 0.0,
                               "reason": "not placed by the judge", "source": source})
        _disagreement_guard(c, groups, before)
    for g in groups:
        g["type"] = cands_type(cands, g["members"][0])
    groups = guard_groups(groups, eps)
    for i, g in enumerate(groups):
        g["group_id"] = f"G{i + 1:03d}"
    return groups


def guard_groups(groups: list[dict], eps) -> list[dict]:
    """Code checks on the judge's verdicts.
    1. A verdict of 'inconsistent' whose own reason says 'disguise', 'deliberate', 'alias'... goes to review.
    2. Two spellings used together in one line are different entities, so they are split out of the group.
    3. For speaker labels the canonical label is the most-used one (fewest edits for production)."""
    units = _units(eps)
    inv = rules.speaker_inventory(eps)
    out = []
    for g in groups:
        g = dict(g)
        if g["verdict"] == "inconsistent" and CONTRADICTION_WORDS.search(g.get("reason", "")):
            g["verdict"] = "uncertain"
            g["reason"] = "[code check: verdict contradicts its own reason] " + g["reason"]
        if g["type"] == "NAME" and g["verdict"] == "inconsistent" and len(g["members"]) > 1:
            can_units = {u.id for u in occurrences(units, g["canonical"])}
            evicted = [m for m in g["members"] if m != g["canonical"]
                       and any(u.id in can_units for u in occurrences(units, m))
                       and not (g["canonical"] in m or m in g["canonical"])]
            for m in evicted:
                g["members"] = [x for x in g["members"] if x != m]
                out.append({**g, "members": [m], "canonical": m, "verdict": "distinct",
                            "reason": f"[code check] used together with {g['canonical']} in the same line, so a different entity"})
        if g["type"] == "NAME" and g["verdict"] == "inconsistent" and len(g["members"]) > 1:
            unlike = [m for m in g["members"] if m != g["canonical"] and rules.name_similarity(m, g["canonical"]) < 0.6]
            for m in unlike:  # drift keeps some resemblance; a completely different name needs a human
                g["members"] = [x for x in g["members"] if x != m]
                out.append({**g, "members": [g["canonical"], m], "verdict": "uncertain", "confidence": 0.5,
                            "reason": f"[code check: AI merged dissimilar names {m} / {g['canonical']}] " + g["reason"]})
        if g["type"] == "SPEAKER" and g["verdict"] == "inconsistent" and all(m in inv for m in g["members"]):
            top = max(g["members"], key=lambda m: (inv[m]["lines"], m))
            if top != g["canonical"]:
                g["reason"] = f"[code rule: canonical = most-used label {top}] " + g["reason"]
                g["canonical"] = top
        out.append(g)
    return out


def _disagreement_guard(c: dict, groups: list[dict], start: int) -> None:
    """The AI may add findings or explain them away, but it may not silently delete what code found.
    If code linked spellings (or typo-like speaker labels) and the AI found nothing inconsistent,
    those members go to 'needs review' with both views stated."""
    mine = groups[start:]
    if c["type"] == "SPEAKER":
        labs = c["members"]
        typo = {x for i, a in enumerate(labs) for b in labs[i + 1:]
                if len(a) > 3 and len(b) > 3 and JaroWinkler.normalized_similarity(a, b) >= 0.9 for x in (a, b)}
        if len(typo) >= 2 and not any(g["verdict"] in ("inconsistent", "deliberate") and len(set(g["members"]) & typo) > 1 for g in mine):
            for g in mine:
                g["members"] = [m for m in g["members"] if m not in typo]
            groups[start:] = [g for g in mine if g["members"]]
            ordered = [m for m in labs if m in typo]
            groups.append({"members": ordered, "canonical": ordered[0], "verdict": "uncertain", "confidence": 0.5,
                           "reason": "code and AI disagree: labels are spelled almost the same, the AI called them different people; check by hand",
                           "source": "code vs AI"})
        return
    if c["type"] != "NAME":
        return
    if any(g["verdict"] == "inconsistent" and len(g["members"]) > 1 for g in mine):
        return
    names = c["members"]
    close = {x for a, b, _ in rules.spelling_links(names) for x in (a, b)}
    if len(close) < 2:
        return
    for g in mine:
        g["members"] = [m for m in g["members"] if m not in close]
    groups[start:] = [g for g in mine if g["members"]]
    ordered = [m for m in names if m in close]
    groups.append({"members": ordered, "canonical": ordered[0], "verdict": "uncertain", "confidence": 0.5,
                   "reason": "code and AI disagree: code sees spelling variants, the AI cleared them; check by hand",
                   "source": "code vs AI"})


def cands_type(cands, member):
    for c in cands:
        if member in c["members"]:
            return c["type"]
    return "NAME"


def _code_verdict(c: dict, eps) -> list[dict]:
    if c["type"] == "NAME":
        names = c["members"]
        best = max((rules.name_similarity(a, b) for i, a in enumerate(names) for b in names[i + 1:]), default=0)
        conf = 0.75 if best >= 0.9 else 0.55
        return [{"members": names, "canonical": names[0], "verdict": "inconsistent" if conf >= 0.7 else "uncertain",
                 "confidence": conf, "reason": f"spelling similarity up to {best:.2f} (code only)", "source": "code"}]
    return [{"members": c["members"], "canonical": c["members"][0], "verdict": "uncertain", "confidence": 0.55,
             "reason": "label overlap after removing titles (code only, cannot tell deliberate from mistake)", "source": "code"}]


def trait_groups_from_decisions(cands: list[dict], decisions: dict, mode: str, source: str = "ai") -> list[dict]:
    out = []
    for c in cands:
        d = (decisions or {}).get(c["id"])
        counts = {v: len(lst) for v, lst in c["values"].items()}
        majority = max(counts, key=lambda v: (counts[v], v))
        if d is None:
            out.append({"type": "TRAIT", "character": c["character"], "trait": c["trait"], "canonical": majority,
                        "values_to_fix": [v for v in counts if v != majority], "verdict": "inconsistent", "confidence": 0.55,
                        "reason": "different values (code only)",
                        "source": "code" if mode == "code" else "code (no AI answer)", "cand": c["id"], "_values": c["values"]})
            continue
        verdict = str(d.get("verdict", "uncertain")).lower()
        if verdict not in {"inconsistent", "deliberate", "compatible", "uncertain"}:
            verdict = "uncertain"
        reason_prefix = ""
        if verdict == "compatible" and _clearly_different(c["trait"], set(counts)):
            verdict, reason_prefix = "uncertain", "[code check: values clearly differ, so not cleared] "
        can = _norm_value(c["trait"], str(d.get("canonical_value", ""))) or majority
        if can not in counts:
            can = majority
        fix = [_norm_value(c["trait"], str(v)) for v in (d.get("values_to_fix") or [])]
        fix = [v for v in fix if v in counts and v != can] or [v for v in counts if v != can]
        try:
            conf = float(d.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        out.append({"type": "TRAIT", "character": c["character"], "trait": c["trait"], "canonical": can, "values_to_fix": fix,
                    "verdict": verdict, "confidence": round(conf, 2), "reason": reason_prefix + str(d.get("reason", ""))[:300],
                    "source": source, "cand": c["id"], "_values": c["values"]})
    return out


def _clearly_different(trait: str, values: set[str]) -> bool:
    """True when no two values share a colour word (brown vs hazel), or scar vs no scar, or different item materials."""
    if trait in ("eye_colour", "hair_colour"):
        words = [set(v.replace("-", " ").split()) for v in values]
        return any(not (a & b) for i, a in enumerate(words) for b in words[i + 1:])
    if trait.startswith("scar"):
        return "no scar" in values
    if trait.startswith("signature"):
        return _conflicting(trait, values)
    return False


def identity_map_from_groups(groups: list[dict], inv: dict) -> dict[str, str]:
    """Map every speaker label (and its core name) to one person name."""
    imap = {}
    for lb in inv:
        core = rules.strip_titles(lb).title()
        imap[lb] = core or lb.title()
        if core:
            imap.setdefault(core.upper(), core)
    for g in groups:
        if g.get("type") == "SPEAKER" and g["verdict"] in {"inconsistent", "deliberate"}:
            person = rules.strip_titles(g["canonical"]).title() or g["canonical"].title()
            for m in g["members"]:
                imap[m] = person
                core = rules.strip_titles(m)
                if core and core != m and (core not in inv or core in g["members"]):
                    imap[core] = person
    return imap


# ============================================================================= flags
def make_flags(eps: list[Episode], groups: list[dict], trait_groups: list[dict], inv: dict, threshold: float) -> list[dict]:
    units = _units(eps)
    common = {w for e in eps for w in re.findall(r"\b[a-z]+\b", e.raw)}
    for g in groups:
        if g["type"] == "SPEAKER":
            words = {w.lower() for m in g["members"] for w in rules.strip_titles(m).split()}
            g["generic_role"] = bool(words) and words <= common
    uidx = {u.id: u for u in units}
    flags = []

    def status_of(g):
        if g["verdict"] == "uncertain":
            return "needs_review"
        return "flag" if g["confidence"] >= threshold else "needs_review"

    def add(g, u, found, canonical, fix, severity=None, status=None, ftype=None):
        flags.append({
            "flag_id": f"F{len(flags) + 1:04d}", "group_id": g.get("group_id", g.get("cand", "")),
            "type": ftype or g["type"], "verdict": g["verdict"], "status": status or status_of(g),
            "severity": severity or SEVERITY.get(u.kind, "Medium"), "episode": u.ep, "block": u.id, "where": u.where,
            "kind": u.kind, "speaker": u.speaker, "found": found, "canonical": canonical,
            "snippet": snippet(u.text, found if ftype != "SPEAKER" else "", 60) if found else u.text[:120],
            "fix": fix, "confidence": g["confidence"], "source": g["source"], "reason": g["reason"]})

    for g in groups:
        if g["type"] == "NAME" and len(g["members"]) > 1 and g["verdict"] in {"inconsistent", "uncertain"}:
            for m in g["members"]:
                if m == g["canonical"] or rules.is_subphrase(m, g["canonical"]):
                    continue  # the same name written shorter or longer is not an error
                occ = occurrences(units, m)
                if g["verdict"] == "uncertain":
                    occ = occ[:1]  # one review row per spelling, not per mention
                for u in occ:
                    add(g, u, m, g["canonical"], f'replace "{m}" with "{g["canonical"]}"')
        elif g["type"] == "SPEAKER" and len(g["members"]) > 1:
            for lb in g["members"]:
                if lb == g["canonical"] or g["verdict"] == "distinct":
                    continue
                by_ep = defaultdict(list)
                for e in eps:
                    for b in e.blocks:
                        if b.speaker == lb:
                            by_ep[e.num].append(b)
                for epn, bl in sorted(by_ep.items()):
                    if g["verdict"] == "deliberate":
                        add(g, bl[0], lb, g["canonical"], f"keep label on screen; map {lb} to the voice/face of {g['canonical']} ({len(bl)} lines)",
                            severity="Info", status="info", ftype="SPEAKER")
                    else:
                        add(g, bl[0], lb, g["canonical"], f"rename speaker label {lb} -> {g['canonical']} ({len(bl)} line{'s' if len(bl) != 1 else ''} in Ep {epn})",
                            severity="Low" if g.get("generic_role") else "Medium", ftype="SPEAKER")
                    if g["verdict"] == "uncertain":
                        break
    for tg in trait_groups:
        if tg["verdict"] not in {"inconsistent", "uncertain"}:
            continue
        cand_vals = tg.get("_values", {})
        for v in tg["values_to_fix"]:
            for ex in cand_vals.get(v, []):
                u = uidx[ex["block"]]
                add(tg, u, ex["as_written"], tg["canonical"],
                    f'{tg["character"]}: change "{v}" to "{tg["canonical"]}" ({tg["trait"].replace("_", " ")})', ftype="TRAIT")
    return flags


# ============================================================================= human decisions
def apply_human(groups: list[dict], trait_groups: list[dict], decisions: dict, eps, trait_cands: list[dict]) -> tuple[list, list]:
    units = _units(eps)
    groups = [dict(g, members=list(g["members"])) for g in groups]
    for spec in (decisions.get("names") or []):
        if "canonical" in spec:
            mem = [spec["canonical"]] + [m for m in spec.get("same_as", []) if m != spec["canonical"]]
            mem = [m for m in mem if occurrences(units, m)]
            _carve(groups, mem)
            groups.append({"type": "NAME", "members": mem, "canonical": spec["canonical"], "verdict": "inconsistent",
                           "confidence": 1.0, "reason": spec.get("note", "human decision"), "source": "human"})
        for m in spec.get("keep_separate", []) or []:
            _carve(groups, [m])
            groups.append({"type": "NAME", "members": [m], "canonical": m, "verdict": "distinct", "confidence": 1.0,
                           "reason": spec.get("note", "human decision: separate entity"), "source": "human"})
    for spec in (decisions.get("speakers") or []):
        mem = [spec["canonical"]] + [m for m in spec.get("same_person", []) if m != spec["canonical"]]
        _carve(groups, mem)
        groups.append({"type": "SPEAKER", "members": mem, "canonical": spec["canonical"],
                       "verdict": "deliberate" if spec.get("deliberate") else "inconsistent", "confidence": 1.0,
                       "reason": spec.get("note", "human decision"), "source": "human"})
    trait_groups = [dict(t) for t in trait_groups]
    for spec in (decisions.get("traits") or []):
        who, trait = spec["character"], spec["trait"]
        match = [t for t in trait_groups if t["character"].lower() == who.lower() and t["trait"] == trait]
        cand = next((c for c in trait_cands if c["character"].lower() == who.lower() and c["trait"] == trait), None)
        for t in match:
            trait_groups.remove(t)
        if cand:
            can = _norm_value(trait, spec["canonical_value"]) or spec["canonical_value"]
            trait_groups.append({"type": "TRAIT", "character": cand["character"], "trait": trait, "canonical": can,
                                 "values_to_fix": [v for v in cand["values"] if v != can], "verdict": "inconsistent",
                                 "confidence": 1.0, "reason": spec.get("note", "human decision"), "source": "human",
                                 "cand": cand["id"], "_values": cand["values"]})
    groups = [g for g in groups if g["members"]]
    for i, g in enumerate(groups):
        g["group_id"] = f"G{i + 1:03d}"
    return groups, trait_groups


def _carve(groups: list[dict], names: list[str]) -> None:
    s = set(names)
    for g in groups:
        if s & set(g["members"]):
            g["members"] = [m for m in g["members"] if m not in s]
            if g["canonical"] in s and g["members"]:
                g["canonical"] = g["members"][0]


# ============================================================================= fixes
def patch_scripts(eps: list[Episode], groups: list[dict], trait_groups: list[dict], only_human: bool = True) -> dict:
    """Apply accepted fixes to the original script text. Returns {source_file: (original, patched)}."""
    ok = (lambda g: g["source"] == "human") if only_human else (lambda g: g["verdict"] == "inconsistent")
    name_fix = [(m, g["canonical"]) for g in groups if g["type"] == "NAME" and g["verdict"] == "inconsistent" and ok(g)
                for m in g["members"] if m != g["canonical"]]
    label_fix = [(m, g["canonical"]) for g in groups if g["type"] == "SPEAKER" and g["verdict"] == "inconsistent" and ok(g)
                 for m in g["members"] if m != g["canonical"]]
    trait_fix = defaultdict(list)  # episode -> [(block_line, old_colour, new_colour)]
    uidx = {u.id: u for e in eps for u in e.blocks}
    for tg in trait_groups:
        if tg["verdict"] == "inconsistent" and ok(tg):
            for v in tg["values_to_fix"]:
                for ex in tg.get("_values", {}).get(v, []):
                    u = uidx.get(ex["block"])
                    old_phrase = rules.colour_phrase(ex["as_written"]) if tg["trait"] in ("eye_colour", "hair_colour") else None
                    if u:
                        trait_fix[u.ep].append((u.line, old_phrase or v, tg["canonical"]))
    out = {}
    for e in eps:
        new = e.raw
        for old, can in name_fix:
            new = word_re(old).sub(can, new)
        for old, can in label_fix:
            new = re.sub(rf"^\*\*{re.escape(old)}(?=\s*\(|:\*\*)", f"**{can}", new, flags=re.M)
        if trait_fix.get(e.num):
            lines = new.split("\n")
            for line_no, old, can in trait_fix[e.num]:
                for k in range(line_no - 1, min(len(lines), line_no + 8)):
                    if re.search(rf"\b{re.escape(old)}\b", lines[k], re.I):
                        lines[k] = re.sub(rf"\b{re.escape(old)}\b", can, lines[k], count=1, flags=re.I)
                        break
            new = "\n".join(lines)
        if new != e.raw:
            out[e.source] = (e.raw, new)
    return out


def unified_diff(patches: dict) -> str:
    chunks = []
    for src, (a, b) in sorted(patches.items()):
        chunks.extend(difflib.unified_diff(a.split("\n"), b.split("\n"), fromfile=f"original/{src}", tofile=f"fixed/{src}", lineterm="", n=0))
    return "\n".join(chunks) + ("\n" if chunks else "")


# ============================================================================= series bible
def build_bible(eps, groups, trait_groups, inv, reader) -> dict:
    people = {}
    imap = identity_map_from_groups(groups, inv)
    for lb, d in inv.items():
        person = imap.get(lb, lb.title())
        p = people.setdefault(person, {"labels": {}, "lines": 0, "episodes": set(), "traits": {}})
        p["labels"][lb] = d["lines"]
        p["lines"] += d["lines"]
        p["episodes"].update(d["episodes"])
    alias_note = {}
    for g in groups:
        if g["type"] == "SPEAKER" and g["verdict"] in {"inconsistent", "deliberate"}:
            for m in g["members"]:
                alias_note[m] = "use canonical label" if g["verdict"] == "inconsistent" else "deliberate on-screen alias"
    trait_votes = defaultdict(Counter)
    for t in (reader or {}).get("traits", []):
        trait = "eye_colour" if t["trait"].startswith("eye") else ("hair_colour" if t["trait"] == "hair" else t["trait"])
        nv = _norm_value(trait, t["value"])
        if nv:
            trait_votes[(resolve_person(t["character"], imap), trait)][nv] += 1
    for (who, trait), cnt in trait_votes.items():
        if who in people:
            people[who]["traits"][trait] = cnt.most_common(1)[0][0]
    for tg in trait_groups:
        if tg["verdict"] in {"inconsistent"} and tg["character"] in people:
            people[tg["character"]]["traits"][tg["trait"]] = tg["canonical"]
    names = []
    for g in groups:
        if g["type"] == "NAME" and g["verdict"] == "inconsistent" and len(g["members"]) > 1:
            names.append({"canonical": g["canonical"], "do_not_use": [m for m in g["members"] if m != g["canonical"]], "source": g["source"]})
        elif g["type"] == "NAME" and g["verdict"] == "distinct" and g["source"] == "human":
            names.append({"canonical": g["canonical"], "do_not_use": [], "note": "separate entity (confirmed)", "source": "human"})
    chars = []
    for person, p in sorted(people.items(), key=lambda kv: -kv[1]["lines"]):
        chars.append({"name": person, "speaker_labels": [{"label": lb, "lines": n, "note": alias_note.get(lb, "")} for lb, n in sorted(p["labels"].items(), key=lambda kv: -kv[1])],
                      "lines": p["lines"], "episodes": ep_ranges(p["episodes"]), "traits": p["traits"]})
    return {"characters": chars, "names": names}


def bible_prompt_block(bible: dict, max_chars: int = 2500) -> str:
    lines = ["SERIES CONTINUITY SHEET - use these exact names and traits; do not invent variants."]
    for n in bible["names"]:
        if n["do_not_use"]:
            lines.append(f"- Always write \"{n['canonical']}\" (never: {', '.join(n['do_not_use'])}).")
    for c in bible["characters"][:15]:
        alias = [s["label"] for s in c["speaker_labels"] if s["note"]]
        tr = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in c["traits"].items())
        if alias or tr:
            extra = f" Speaker labels used: {', '.join(alias)}." if alias else ""
            lines.append(f"- {c['name']}:{extra}{' ' + tr + '.' if tr else ''}")
    return "\n".join(lines)[:max_chars]
