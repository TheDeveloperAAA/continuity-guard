"""Proof that it works: planted errors (tuning set + frozen test set), golden set, baseline, clean control.

Planted errors are chosen by a seeded random script from the show's own text, before any tuning.
One error type (relationship flips) is deliberately out of scope, to show the tool's limits honestly.
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
from collections import defaultdict

from . import ai, rules
from .issues import ITEM_DESC, ITEM_NOUNS, word_re
from .parser import Episode, parse_episode

COLOUR_SWAP = {"blue": "brown", "green": "brown", "grey": "brown", "gray": "brown", "brown": "blue", "hazel": "blue",
               "dark": "blond", "blond": "dark", "blonde": "dark", "golden": "dark", "silver": "auburn", "black": "blond",
               "red": "black", "auburn": "black", "amber": "blue", "white": "dark", "raven": "blond", "chestnut": "silver"}
DESC_SWAP = {"emerald": "ruby", "ruby": "emerald", "sapphire": "emerald", "gold": "silver", "golden": "silver",
             "silver": "gold", "oval": "square", "square": "oval", "round": "square", "pearl": "jade", "jade": "pearl",
             "diamond": "pearl", "iron": "brass", "brass": "iron", "copper": "silver", "bronze": "silver"}
REL_SWAP = {"sister": "cousin", "brother": "cousin", "mother": "aunt", "father": "uncle", "stepmother": "aunt",
            "husband": "brother", "wife": "sister", "son": "nephew", "daughter": "niece"}
TYPES = ["PERSON_NAME", "PLACE_OR_FAMILY_NAME", "SPEAKER_LABEL", "COLOUR", "ITEM_OR_SCAR", "RELATIONSHIP_OUT_OF_SCOPE"]
SUFFIXES = ["ton", "ford", "wood", "by", "field", "ley", "ham", "worth", "combe", "dale"]


def _variant(name: str, rng: random.Random, existing: set[str], style: str) -> str | None:
    cands = set()
    n = name
    if style == "suffix" and len(n) >= 6:
        for k in (2, 3, 4):
            for s in SUFFIXES:
                v = n[:-k] + s
                if v != n:
                    cands.add(v)
    vowels = "aeiou"
    for i in range(1, len(n) - 1):
        c = n[i]
        if c.lower() in vowels:
            for v in vowels:
                if v != c.lower():
                    cands.add(n[:i] + (v if c.islower() else v.upper()) + n[i + 1:])
        elif c.isalpha():
            cands.add(n[:i] + c + n[i:])  # doubled letter
    cands = [c for c in cands if c not in existing and c.lower() != n.lower() and c[0] == n[0]]
    if style != "suffix":
        cands = [c for c in cands if rules.name_similarity(c, n) >= 0.85]
    cands.sort()
    return rng.choice(cands) if cands else None


def _edit_block_line(ep: Episode, block, pattern: re.Pattern, repl: str) -> str | None:
    """Replace the first match of pattern in the source lines of one block. Returns new raw or None."""
    lines = ep.raw.split("\n")
    for k in range(max(0, block.line - 1), min(len(lines), block.line + 8)):
        if k > block.line - 1 and lines[k].startswith("**"):
            break
        if pattern.search(lines[k]) and not lines[k].startswith("Start:"):
            if lines[k].startswith("**") and k == block.line - 1 and block.kind == "DIALOGUE" and not pattern.pattern.startswith("^"):
                continue
            lines[k] = pattern.sub(repl, lines[k], count=1)
            return "\n".join(lines)
    return None


def plant_errors(eps: list[Episode], seed: int, allowed: set[int], per_type: int) -> tuple[list[Episode], list[dict]]:
    rng = random.Random(seed)
    eps = [copy.deepcopy(e) for e in eps]
    by_num = {e.num: e for e in eps}
    inv = rules.speaker_inventory(eps)
    existing_names = set(rules.proper_names(eps))
    corpus_words = {w for e in eps for w in re.findall(r"[A-Za-z]+", e.raw)}
    lower_words = set(re.findall(r"\b[a-z]+\b", "\n".join(e.raw for e in eps)))
    min_lines = 15 if sum(d["lines"] for d in inv.values()) > 300 else 3
    person_names = sorted({rules.strip_titles(lb).title() for lb, d in inv.items() if d["lines"] >= min_lines and len(rules.strip_titles(lb).split()) == 1
                           and rules.strip_titles(lb).lower() not in lower_words})
    cluster_members = {x for a, b, _ in rules.spelling_links(sorted(existing_names)) for x in (a, b)}
    occ = rules.proper_names(eps)
    label_tokens = {t.title() for lb in inv for t in re.split(r"[\s']+", lb) if t}
    blob = "\n".join(e.raw for e in eps)

    def place_like(n):
        return re.search(rf"\b(?:Lord|Lady|Duke of|Duchess of|House of|Earl of)\s+{re.escape(n)}\b|\b{re.escape(n)}\s+(?:Manor|Estate|Hall|House|Abbey|Castle|family|Village|Wood|Woods|Court|Street|Road)\b", blob)
    family_place = sorted(n for n, ids in occ.items() if len(ids) >= 2 and n not in person_names and n not in cluster_members
                          and place_like(n))
    used_blocks: set[str] = set()
    muts: list[dict] = []

    def blocks_in(kinds=("DIALOGUE", "ACTION", "CAMERA")):
        return [b for e in eps if e.num in allowed for b in e.blocks if b.kind in kinds and b.id not in used_blocks]

    def commit(ep, block, new_raw, mtype, old, new, note=""):
        ep2 = parse_episode(new_raw, ep.source)
        by_num[ep.num].__dict__.update(ep2.__dict__)
        used_blocks.add(block.id)
        muts.append({"id": f"M{len(muts) + 1:02d}", "type": mtype, "episode": ep.num, "block": block.id, "kind": block.kind,
                     "original": old, "planted": new, "note": note})

    for mtype in TYPES:
        made, tries = 0, 0
        used_names: set[str] = set()
        while made < per_type and tries < 200:
            tries += 1
            if mtype in ("PERSON_NAME", "PLACE_OR_FAMILY_NAME"):
                pool = person_names if mtype == "PERSON_NAME" else family_place
                if not pool:
                    break
                pool_left = [p for p in pool if p not in used_names] or pool
                name = rng.choice(pool_left)
                cands = [b for b in blocks_in() if word_re(name).search(b.text)]
                if not cands:
                    continue
                b = rng.choice(cands)
                var = _variant(name, rng, existing_names | corpus_words, "typo" if mtype == "PERSON_NAME" else "suffix")
                if not var:
                    continue
                new_raw = _edit_block_line(by_num[b.ep], b, word_re(name), var)
                if new_raw:
                    existing_names.add(var)
                    used_names.add(name)
                    commit(by_num[b.ep], b, new_raw, mtype, name, var)
                    made += 1
            elif mtype == "SPEAKER_LABEL":
                labs = [lb for lb, d in inv.items() if d["lines"] >= min_lines and len(lb.split()) == 1 and lb.lower() not in lower_words]
                if not labs:
                    break
                lb = rng.choice([x for x in labs if x not in used_names] or labs)
                cands = [b for b in blocks_in(("DIALOGUE",)) if b.speaker == lb]
                if not cands:
                    continue
                b = rng.choice(cands)
                var = _variant(lb.title(), rng, existing_names | corpus_words, "typo")
                if not var:
                    continue
                var = var.upper()
                new_raw = _edit_block_line(by_num[b.ep], b, re.compile(rf"^\*\*{re.escape(lb)}(?=\s*\(|:\*\*)"), f"**{var}")
                if new_raw:
                    used_names.add(lb)
                    commit(by_num[b.ep], b, new_raw, mtype, lb, var)
                    made += 1
            elif mtype == "COLOUR":
                mentions = rules.trait_mentions_code(eps)
                by_char = defaultdict(set)
                for t in mentions:
                    by_char[(t["character"], t["trait"])].add(t["block"])
                cands = [t for t in mentions if t["block"] not in used_blocks and int(t["block"][1:3]) in allowed
                         and len(by_char[(t["character"], t["trait"])]) >= 2]
                if not cands:
                    break
                t = rng.choice(cands)
                col = t["phrase"].split()[0].split("-")[0].lower()
                if col not in COLOUR_SWAP:
                    continue
                b = next(x for x in by_num[int(t["block"][1:3])].blocks if x.id == t["block"])
                new_phrase = re.sub(rf"^{re.escape(t['phrase'].split()[0])}", COLOUR_SWAP[col], t["phrase"], flags=re.I)
                new_raw = _edit_block_line(by_num[b.ep], b, re.compile(re.escape(t["phrase"])), new_phrase)
                if new_raw:
                    commit(by_num[b.ep], b, new_raw, mtype, t["phrase"], new_phrase, f"{t['character']} {t['trait']}")
                    made += 1
            elif mtype == "ITEM_OR_SCAR":
                rx = re.compile(rf"\b({'|'.join(sorted(ITEM_DESC & set(DESC_SWAP)))})\s+(?:\w+\s+)?({'|'.join(sorted(ITEM_NOUNS))})\b", re.I)
                cands = []
                for b in blocks_in(("CAMERA", "ACTION", "DIALOGUE")):
                    for m in rx.finditer(b.text):
                        same = [x for e in eps for x in e.blocks if x.id != b.id and re.search(rf"\b{re.escape(m.group(0))}\b", x.text, re.I)]
                        if same:
                            cands.append((b, m.group(0), m.group(1)))
                    if re.search(r"\bscarred\b", b.text) and b.kind != "DIALOGUE":
                        cands.append((b, "scarred", "scarred"))
                if not cands:
                    break
                b, phrase, desc = rng.choice(cands)
                if desc == "scarred":
                    new_phrase = "smooth, unscarred"
                else:
                    new_phrase = re.sub(rf"^{re.escape(desc)}", DESC_SWAP[desc.lower()], phrase, flags=re.I)
                new_raw = _edit_block_line(by_num[b.ep], b, re.compile(rf"\b{re.escape(phrase)}\b"), new_phrase)
                if new_raw:
                    commit(by_num[b.ep], b, new_raw, mtype, phrase, new_phrase)
                    made += 1
            elif mtype == "RELATIONSHIP_OUT_OF_SCOPE":
                rx = re.compile(r"\b(her|his|my|your)\s+(" + "|".join(REL_SWAP) + r")\b")
                cands = [(b, m) for b in blocks_in() for m in rx.finditer(b.text)]
                if not cands:
                    break
                b, m = rng.choice(cands)
                new = f"{m.group(1)} {REL_SWAP[m.group(2)]}"
                new_raw = _edit_block_line(by_num[b.ep], b, re.compile(rf"\b{re.escape(m.group(0))}\b"), new)
                if new_raw:
                    commit(by_num[b.ep], b, new_raw, mtype, m.group(0), new)
                    made += 1
    return [by_num[k] for k in sorted(by_num)], muts


# ------------------------------------------------------------------------------------------ scoring
def caught_by_pipeline(m: dict, res: dict, use_final: bool = False) -> bool:
    flags = res["final_flags"] if use_final else res["flags"]
    groups = res["final_groups"] if use_final else res["groups"]
    if m["type"] in ("PERSON_NAME", "PLACE_OR_FAMILY_NAME"):
        if any(f["block"] == m["block"] and f["found"] == m["planted"] for f in flags):
            return True
        return any(g["type"] == "NAME" and g["verdict"] in {"inconsistent", "uncertain"} and {m["original"], m["planted"]} <= set(g["members"]) for g in groups)
    if m["type"] == "SPEAKER_LABEL":
        return any(f["type"] == "SPEAKER" and f["found"] == m["planted"] for f in flags)
    if m["type"] in ("COLOUR", "ITEM_OR_SCAR"):
        return any(f["block"] == m["block"] and f["type"] == "TRAIT" for f in flags)
    word = m["planted"].split()[-1].lower()
    return any(f["block"] == m["block"] and word in (f["found"] or "").lower() for f in flags)


def caught_by_baseline(m: dict, issues: list[dict]) -> bool:
    key = m["planted"].lower()
    for it in issues:
        text = f"{it.get('quote', '')} {it.get('issue', '')}".lower()
        same_place = str(it.get("block", "")) == m["block"] or str(it.get("episode", "")) == str(m["episode"])
        if same_place and key in text:
            return True
    return False


def score(muts: list[dict], caught_fn) -> dict:
    per = defaultdict(lambda: [0, 0])
    detail = []
    for m in muts:
        c = bool(caught_fn(m))
        per[m["type"]][0] += int(c)
        per[m["type"]][1] += 1
        detail.append({**m, "caught": c})
    in_scope = [d for d in detail if d["type"] != "RELATIONSHIP_OUT_OF_SCOPE"]
    return {"per_type": {k: {"caught": v[0], "planted": v[1]} for k, v in per.items()},
            "in_scope_recall": round(sum(d["caught"] for d in in_scope) / max(1, len(in_scope)), 3),
            "in_scope_caught": sum(d["caught"] for d in in_scope), "in_scope_planted": len(in_scope), "detail": detail}


# ------------------------------------------------------------------------------------------ golden set
def golden_score_pipeline(golden: dict, res: dict) -> dict:
    groups, tgroups = res["groups"], res["trait_groups"]
    house = list(golden["name_group"]["spellings"])
    live = [g for g in groups if g["type"] == "NAME" and g["verdict"] in {"inconsistent", "uncertain"}]
    best = max((set(g["members"]) & set(house) for g in live), key=len, default=set())
    items = [{"item": f"house spelling {s}", "hit": s in best and len(best) >= 2} for s in house]
    for a, b in golden["speaker_pairs"]:
        hit = any(g["type"] == "SPEAKER" and g["verdict"] in {"inconsistent", "deliberate", "uncertain"} and {a, b} <= set(g["members"]) for g in groups)
        items.append({"item": f"speaker {a} = {b}", "hit": hit})
    for sp in golden.get("spelling_pairs", []):
        hit = any(g["type"] == "NAME" and g["verdict"] in {"inconsistent", "uncertain"} and {sp["canonical"], sp["variant"]} <= set(g["members"]) for g in groups)
        items.append({"item": f"spelling {sp['variant']} -> {sp['canonical']}", "hit": hit})
    for t in golden["traits"]:
        vals = set(t["values"])
        hit = any(tg["trait"] == "eye_colour" and tg["verdict"] in {"inconsistent", "uncertain"} and vals <= set(tg.get("_values", {})) for tg in tgroups)
        items.append({"item": f"{t['character']} eye colour {'/'.join(sorted(vals))}", "hit": hit})
    return {"recall": round(sum(i["hit"] for i in items) / len(items), 3), "hits": sum(i["hit"] for i in items), "items_total": len(items), "items": items}


def golden_score_baseline(golden: dict, issues: list[dict]) -> dict:
    texts = [f"{it.get('quote', '')} {it.get('issue', '')}" for it in issues]
    blob = " ".join(texts)
    items = [{"item": f"house spelling {s}", "hit": bool(re.search(rf"\b{s}\b", blob))} for s in golden["name_group"]["spellings"]]
    for a, b in golden["speaker_pairs"]:
        items.append({"item": f"speaker {a} = {b}", "hit": any(a.lower() in t.lower() and b.lower() in t.lower() for t in texts)})
    for sp in golden.get("spelling_pairs", []):
        items.append({"item": f"spelling {sp['variant']} -> {sp['canonical']}", "hit": sp["variant"] in blob})
    for t in golden["traits"]:
        items.append({"item": f"{t['character']} eye colour", "hit": all(v in blob for v in t["values"])})
    return {"recall": round(sum(i["hit"] for i in items) / len(items), 3), "hits": sum(i["hit"] for i in items), "items_total": len(items), "items": items}


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)


def run_baseline_and_save(eps, llm, prompts, cfg, out_dir) -> dict:
    res = ai.run_baseline(eps, llm, prompts, cfg)
    write_json(os.path.join(out_dir, "baseline_issues.json"), res)
    return res
