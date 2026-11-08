"""Offline stand-in for the model, used only to test the plumbing without a key.

It answers in the same JSON shapes as the real model, using simple rules. Its answers are not a quality claim.
"""
from __future__ import annotations

import json
import os
import threading

from . import rules
from .llm import Usage


class MockLLM:
    def __init__(self, raw_dir: str | None = None):
        self.usage = Usage()
        self.raw_dir = raw_dir
        self._n = 0
        self._lock = threading.Lock()
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)

    def chat_json(self, system: str, user: str, stage: str = "call", payload=None) -> dict:
        with self._lock:
            self._n += 1
            n = self._n
        payload = payload or {}
        if stage.startswith("reader"):
            out = self._reader(payload["episodes"])
        elif stage.startswith("judge_names"):
            out = self._judge_names(payload["candidates"])
        elif stage.startswith("judge_traits"):
            out = self._judge_traits(payload["candidates"])
        elif stage.startswith("baseline"):
            out = self._baseline(payload["episodes"])
        else:
            out = {}
        with self._lock:
            self.usage.add(stage, len(user) // 4, len(json.dumps(out)) // 4, 0.0, False, "mock")
        if self.raw_dir:
            with open(os.path.join(self.raw_dir, f"{n:03d}_{stage}.json"), "w", encoding="utf-8") as fh:
                json.dump({"model": "mock", "stage": stage, "parsed": out}, fh, indent=1)
        return out

    def _reader(self, eps):
        names, speakers, traits = [], [], []
        occ = rules.proper_names(eps)
        for n, ids in occ.items():
            names.append({"name": n, "kind": "place", "refers_to": "", "block": ids[0], "quote": n})
        for lb, d in rules.speaker_inventory(eps).items():
            core = rules.strip_titles(lb).title() or lb.title()
            speakers.append({"label": lb, "identity": core, "block": d["first_block"], "quote": ""})
        for t in rules.trait_mentions_code(eps):
            traits.append({"character": t["character"], "trait": t["trait"].replace("_colour", "") if t["trait"].startswith("hair") else t["trait"],
                           "value": t["phrase"], "deliberate_change": False, "block": t["block"], "quote": t["phrase"]})
        return {"names": names, "speakers": speakers, "traits": traits}

    def _judge_names(self, cands):
        dec = []
        for c in cands:
            mem = c["members"]
            if c["type"] == "NAME":
                best = max((rules.name_similarity(a, b) for i, a in enumerate(mem) for b in mem[i + 1:]), default=0)
                dec.append({"id": c["id"], "groups": [{"members": mem, "canonical": mem[0],
                            "verdict": "inconsistent" if best >= 0.9 else "uncertain", "confidence": 0.8 if best >= 0.9 else 0.4,
                            "reason": "mock: spelling similarity"}]})
            else:
                dec.append({"id": c["id"], "groups": [{"members": mem, "canonical": mem[0], "verdict": "inconsistent",
                            "confidence": 0.75, "reason": "mock: label overlap"}]})
        return {"decisions": dec}

    def _judge_traits(self, cands):
        return {"decisions": [{"id": c["id"], "verdict": "inconsistent", "canonical_value": "", "values_to_fix": [],
                               "confidence": 0.75, "reason": "mock"} for c in cands]}

    def _baseline(self, eps):
        occ = rules.proper_names(eps)
        issues = []
        for a, b, _ in rules.spelling_links(sorted(occ)):
            minor = a if len(occ[a]) < len(occ[b]) else b
            blk = occ[minor][0]
            issues.append({"episode": int(blk[1:3]), "block": blk, "quote": minor, "issue": f"{minor} vs {a if minor == b else b}"})
        return {"issues": issues}
