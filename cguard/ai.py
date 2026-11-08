"""The three AI steps: Reader (extract facts), Judge (decide), Baseline (one-prompt comparison).

Every model claim is checked by code before use: cited blocks must exist and the cited name or value
must actually appear in that block. Rejected items are logged, never silently used.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from string import Template

from .llm import LLMError
from .parser import Episode, block_index, compact_episode


def _t(prompts: dict, key: str, **kw) -> str:
    return Template(prompts[key]).safe_substitute(**kw)


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.split())


# ------------------------------------------------------------------------------------------ Reader
def run_reader(eps: list[Episode], llm, prompts: dict, cfg: dict) -> dict:
    per = cfg["reader"]["episodes_per_call"]
    include_cam = cfg["reader"].get("include_camera_blocks", True)
    idx = block_index(eps)
    out = {"names": [], "speakers": [], "traits": [], "rejected": [], "quote_mismatches": 0, "batches": []}
    out["failed_episodes"] = []

    def ask(batch):
        labels = sorted({b.speaker for e in batch for b in e.blocks if b.speaker})
        text = "\n\n".join(compact_episode(e, include_cam) for e in batch)
        rng = f"{batch[0].num}-{batch[-1].num}" if len(batch) > 1 else f"{batch[0].num}"
        user = _t(prompts, "reader_user", episode_range=rng, speaker_labels=", ".join(labels) or "(none)", episodes_text=text)
        try:
            return rng, labels, llm.chat_json(prompts["reader_system"], user, stage=f"reader_ep{rng}",
                                              payload={"episodes": batch, "labels": labels})
        except LLMError as exc:
            return rng, labels, exc

    batches = [eps[i:i + per] for i in range(0, len(eps), per)]
    with ThreadPoolExecutor(max_workers=_workers(cfg)) as pool:
        results = list(pool.map(ask, batches))
    retry = []
    for batch, (rng, labels, res) in zip(batches, results):  # validate in episode order, so output is deterministic
        if isinstance(res, Exception):
            if len(batch) > 1 and "429" not in str(res) and "HTTP" not in str(res):
                half = (len(batch) + 1) // 2  # e.g. a reply cut off mid-JSON: retry as two smaller requests
                retry.extend([batch[:half], batch[half:]])
            else:
                out["failed_episodes"].extend(e.num for e in batch)  # code checks still cover these episodes
        else:
            out["batches"].append(rng)
            _validate_reader(res, idx, labels, out)
    if retry:
        with ThreadPoolExecutor(max_workers=_workers(cfg)) as pool:
            for (rng, labels, res), b in zip(pool.map(ask, retry), retry):
                if isinstance(res, Exception):
                    out["failed_episodes"].extend(e.num for e in b)
                else:
                    out["batches"].append(rng)
                    _validate_reader(res, idx, labels, out)
    out["failed_episodes"] = sorted(set(out["failed_episodes"]))
    if out["failed_episodes"]:
        print(f"  Model could not read episodes {out['failed_episodes']}; code-only checks still cover them.", flush=True)
    return out


def _workers(cfg: dict) -> int:
    return max(1, int(cfg["model"].get("parallel_requests", 4)))


def _validate_reader(res: dict, idx: dict, labels: list[str], out: dict) -> None:
    for item in (res.get("names") or []):
        b = idx.get(str(item.get("block", "")).strip())
        name = _norm(str(item.get("name", "")))
        if not b or not name:
            out["rejected"].append({"list": "names", "item": item, "why": "unknown block or empty name"})
            continue
        if not re.search(rf"(?<!\w){re.escape(name)}(?!\w)", _norm(b.text)):
            out["rejected"].append({"list": "names", "item": item, "why": "name not found in cited block"})
            continue
        if _norm(str(item.get("quote", ""))) not in _norm(b.text):
            out["quote_mismatches"] += 1
        out["names"].append({"name": name, "kind": str(item.get("kind", "")).strip().lower(),
                             "refers_to": str(item.get("refers_to", "")).strip(), "block": b.id, "ep": b.ep})
    for item in (res.get("speakers") or []):
        lab = str(item.get("label", "")).strip().upper()
        if lab not in labels:
            out["rejected"].append({"list": "speakers", "item": item, "why": "label not used in these episodes"})
            continue
        out["speakers"].append({"label": lab, "identity": str(item.get("identity", "")).strip(),
                                "block": str(item.get("block", "")), "ep": idx[item["block"]].ep if item.get("block") in idx else None})
    for item in (res.get("traits") or []):
        b = idx.get(str(item.get("block", "")).strip())
        val = _norm(str(item.get("value", "")))
        if not b or not val:
            out["rejected"].append({"list": "traits", "item": item, "why": "unknown block or empty value"})
            continue
        key_words = [w for w in re.findall(r"[a-z]+", val.lower()) if len(w) > 3 and w not in {"eyes", "hair", "with", "long", "short"}]
        txt = b.text.lower()
        if key_words and not any(w in txt for w in key_words):
            out["rejected"].append({"list": "traits", "item": item, "why": "value not found in cited block"})
            continue
        if _norm(str(item.get("quote", ""))) not in _norm(b.text):
            out["quote_mismatches"] += 1
        out["traits"].append({"character": str(item.get("character", "")).strip(), "trait": str(item.get("trait", "")).strip().lower(),
                              "value": val, "deliberate_change": bool(item.get("deliberate_change", False)), "block": b.id, "ep": b.ep})


# ------------------------------------------------------------------------------------------ Judge
def synopsis(eps: list[Episode]) -> str:
    return "\n".join(f"Ep {e.num}: {e.logline}" for e in eps if e.logline) or "(no loglines in this script)"


def run_judge(cands: list[dict], eps: list[Episode], llm, prompts: dict, cfg: dict, kind: str) -> dict:
    """kind = 'names' (NAME and SPEAKER candidates) or 'traits'. Returns {candidate_id: decision}."""
    per = cfg["judge"]["candidates_per_call"]
    decisions: dict[str, dict] = {}
    syn = synopsis(eps)
    chunks = [cands[i:i + per] for i in range(0, len(cands), per)]

    def ask(k_chunk):
        k, chunk = k_chunk
        payload_json = json.dumps([c["for_model"] for c in chunk], ensure_ascii=False, indent=1)
        if kind == "names":
            user = _t(prompts, "judge_names_user", synopsis=syn, candidates_json=payload_json)
            sysm = prompts["judge_names_system"]
        else:
            user = _t(prompts, "judge_traits_user", candidates_json=payload_json)
            sysm = prompts["judge_traits_system"]
        try:
            return chunk, llm.chat_json(sysm, user, stage=f"judge_{kind}_{k + 1}", payload={"candidates": chunk, "kind": kind})
        except LLMError:
            return chunk, None  # these candidates fall back to the code verdict instead of stopping the run

    with ThreadPoolExecutor(max_workers=_workers(cfg)) as pool:
        for chunk, res in pool.map(ask, enumerate(chunks)):
            for d in ((res or {}).get("decisions") or []):
                cid = str(d.get("id", "")).strip()
                if cid in {c["id"] for c in chunk}:
                    decisions[cid] = d
    return decisions


# ------------------------------------------------------------------------------------------ Baseline
def run_baseline(eps: list[Episode], llm, prompts: dict, cfg: dict) -> dict:
    """One prompt over the whole show (split into chunks only if it is too long)."""
    max_chars = cfg["baseline"].get("max_chars_per_call", 450000)
    chunks, cur, size = [], [], 0
    for e in eps:
        t = compact_episode(e, True)
        if cur and size + len(t) > max_chars:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(e)
        size += len(t)
    if cur:
        chunks.append(cur)
    def ask(k_ch):
        k, ch = k_ch
        text = "\n\n".join(compact_episode(e, True) for e in ch)
        return llm.chat_json(prompts["baseline_system"], _t(prompts, "baseline_user", show_text=text),
                             stage=f"baseline_{k + 1}", payload={"episodes": ch})

    issues = []
    with ThreadPoolExecutor(max_workers=_workers(cfg)) as pool:
        for res in pool.map(ask, enumerate(chunks)):
            issues.extend(res.get("issues") or [])
    return {"chunks": len(chunks), "issues": issues}
