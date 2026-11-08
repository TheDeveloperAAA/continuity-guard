"""Orchestrates one ContinuityGuard run: parse -> code checks -> Reader -> candidates -> Judge -> flags -> reports."""
from __future__ import annotations

import datetime as dt
import json
import os

import yaml

from . import ai, issues, report, rules
from .llm import build_llm
from .parser import load_show

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg(path: str | None = None) -> dict:
    with open(path or os.path.join(HERE, "config.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_prompts(path: str | None = None) -> dict:
    with open(path or os.path.join(HERE, "prompts.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_decisions(path: str | None) -> dict | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _dump(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=str)


def run_check(show, out_dir: str, cfg: dict, prompts: dict, mode: str = "full", decisions: dict | None = None,
              llm=None, title: str = "show", write_files: bool = True, eps=None) -> dict:
    """Run the checker on one show. mode: 'full' (code + AI) or 'code' (no model)."""
    os.makedirs(out_dir, exist_ok=True)
    eps = eps if eps is not None else load_show(show)
    if not eps:
        raise ValueError("No episodes found in the input.")
    if mode == "full" and llm is None:
        llm = build_llm(cfg, out_dir)
    t0 = dt.datetime.now()
    thr = cfg["thresholds"]["flag_confidence"]
    parse_summary = {
        "episodes": len(eps), "blocks": sum(len(e.blocks) for e in eps), "timed": all(e.timed for e in eps),
        "speaker_labels": len(rules.speaker_inventory(eps)),
        "runtime_minutes": round(sum(e.duration for e in eps) / 60, 1),
        "episode_numbers": [e.num for e in eps],
        "speech_rate": rules.speech_rate_stats(eps),
    }
    if mode == "full":
        print(f"Reading {len(eps)} episodes with the model ({cfg['reader']['episodes_per_call']} per request)...", flush=True)
    reader = ai.run_reader(eps, llm, prompts, cfg) if mode == "full" else None
    name_c = issues.name_candidates(eps, reader, cfg)
    spk_c, inv = issues.speaker_candidates(eps, reader, cfg)
    cands = name_c + spk_c
    if mode == "full" and cands:
        print(f"Judging {len(cands)} name/speaker candidates...", flush=True)
    dec_names = ai.run_judge(cands, eps, llm, prompts, cfg, "names") if (mode == "full" and cands) else {}
    groups = issues.groups_from_decisions(cands, dec_names, eps, mode, source="ai" if mode == "full" else "code")
    imap = issues.identity_map_from_groups(groups, inv)
    trait_c = issues.trait_candidates(eps, reader, imap)
    dec_traits = ai.run_judge(trait_c, eps, llm, prompts, cfg, "traits") if (mode == "full" and trait_c) else {}
    tgroups = issues.trait_groups_from_decisions(trait_c, dec_traits, mode)
    flags = issues.make_flags(eps, groups, tgroups, inv, thr)
    used = getattr(getattr(llm, "usage", None), "models_used", {}) or {}
    model_label = (", ".join(f"{m} ({n})" for m, n in sorted(used.items(), key=lambda kv: -kv[1])) if used
                   else (getattr(llm, "models", ["mock"])[0] if llm is not None else "none (code only)"))
    meta = {"model": model_label, "mode": mode,
            "episodes": len(eps), "blocks": parse_summary["blocks"], "run_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    if reader:
        meta["reader_rejected"] = len(reader["rejected"])
        meta["reader_items"] = len(reader["rejected"]) + len(reader["names"]) + len(reader["speakers"]) + len(reader["traits"])
    result = {"eps": eps, "reader": reader, "candidates": cands, "trait_candidates": trait_c, "groups": groups,
              "trait_groups": tgroups, "flags": flags, "inv": inv, "meta": meta, "parse_summary": parse_summary}
    final_groups, final_tgroups, final_flags = groups, tgroups, flags
    if decisions:
        final_groups, final_tgroups = issues.apply_human(groups, tgroups, decisions, eps, trait_c)
        final_flags = issues.make_flags(eps, final_groups, final_tgroups, inv, thr)
        result.update({"final_groups": final_groups, "final_trait_groups": final_tgroups, "final_flags": final_flags})
    bible = issues.build_bible(eps, final_groups, final_tgroups, inv, reader)
    result["bible"] = bible
    if write_files:
        _dump(os.path.join(out_dir, "01_parse_summary.json"), parse_summary)
        if reader:
            _dump(os.path.join(out_dir, "02_reader_items.json"), reader)
        _dump(os.path.join(out_dir, "03_candidates.json"), {"names_and_speakers": [{k: v for k, v in c.items() if k != "for_model"} | {"evidence": c["for_model"]} for c in cands],
                                                             "traits": trait_c})
        _dump(os.path.join(out_dir, "04_judge_decisions.json"), {"names_and_speakers": dec_names, "traits": dec_traits})
        r1 = report.write_reports(out_dir, "05_ai", f"{title} (before human decisions)", meta, groups, tgroups, flags)
        result["summary_ai"] = r1["summary"]
        if decisions:
            r2 = report.write_reports(out_dir, "06_final", f"{title} (after human decisions)", meta | {"mode": mode + " + human decisions"},
                                      final_groups, final_tgroups, final_flags)
            result["summary_final"] = r2["summary"]
            patches = issues.patch_scripts(eps, final_groups, final_tgroups, only_human=True)
            pdir = os.path.join(out_dir, "08_fixed_scripts")
            os.makedirs(pdir, exist_ok=True)
            for src, (_, new) in patches.items():
                with open(os.path.join(pdir, src), "w", encoding="utf-8") as fh:
                    fh.write(new)
            with open(os.path.join(out_dir, "08_fixes.diff"), "w", encoding="utf-8") as fh:
                fh.write(issues.unified_diff(patches))
            result["patched_files"] = len(patches)
        report.write_bible(out_dir, "07", bible, issues.bible_prompt_block(bible))
        usage = getattr(llm, "usage", None)
        manifest = {"tool": "ContinuityGuard", "title": title, "meta": meta, "parse_summary": parse_summary,
                    "counts": {"name_candidates": len(name_c), "speaker_candidates": len(spk_c), "trait_candidates": len(trait_c),
                               "groups": len(groups), "flags_ai": len(flags), "flags_final": len(final_flags)},
                    "usage": usage.__dict__ if usage else None, "thresholds": cfg["thresholds"],
                    "model_config": {k: v for k, v in cfg["model"].items() if k != "api_key_env"},
                    "finished_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        _dump(os.path.join(out_dir, "00_run_manifest.json"), manifest)
    return result
