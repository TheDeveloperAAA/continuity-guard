"""Full evaluation suite, run in priority order so the most important results exist even if quota runs out.

1. main      : the show, full mode (+ human decisions if given)   -> golden-set score
2. code      : the show, code-only mode                            -> golden-set score (free)
3. test      : frozen planted-error set, full + code-only          -> catch rate per type
4. baseline  : one-prompt AI on the show and on the test set       -> golden + catch rate
5. dev       : planted tuning set (used only to tune prompts)
6. clean     : clean control script                                 -> false alarms
"""
from __future__ import annotations

import json
import os
import traceback

from . import evaluate as ev
from .llm import LLMError
from .parser import load_show
from .pipeline import run_check

DEV_EPS = set(range(6, 11)) | set(range(31, 36))
TEST_EPS = set(range(11, 16)) | set(range(21, 26)) | set(range(41, 46))


def run_suite(show, out_root: str, cfg: dict, prompts: dict, llm, golden: dict | None = None, decisions: dict | None = None,
              clean_show=None, steps=("main", "code", "test", "baseline", "dev", "clean"), mode: str = "full",
              per_type_test: int = 4, per_type_dev: int = 2, title: str = "show") -> dict:
    os.makedirs(out_root, exist_ok=True)
    summary: dict = {"steps": {}, "errors": {}}
    eps = load_show(show)

    def safe(name, fn):
        if name not in steps:
            return None
        try:
            out = fn()
            summary["steps"][name] = "ok"
            _save(out_root, summary)
            return out
        except (LLMError, Exception) as exc:  # keep going: later steps may still work
            summary["steps"][name] = "failed"
            summary["errors"][name] = f"{type(exc).__name__}: {exc}"[:500]
            with open(os.path.join(out_root, f"error_{name}.txt"), "w") as fh:
                fh.write(traceback.format_exc())
            _save(out_root, summary)
            return None

    def main():
        res = run_check(show, os.path.join(out_root, "1_main"), cfg, prompts, mode=mode, decisions=decisions, llm=llm, title=title, eps=eps)
        out = {"summary_ai": res["summary_ai"], "summary_final": res.get("summary_final"), "patched_files": res.get("patched_files")}
        if golden:
            out["golden"] = ev.golden_score_pipeline(golden, res)
            ev.write_json(os.path.join(out_root, "1_main", "golden_score.json"), out["golden"])
        summary["main"] = {k: (v if k != "golden" else {kk: v[kk] for kk in ("recall", "hits", "items_total")}) for k, v in out.items()}
        return out

    def code():
        res = run_check(show, os.path.join(out_root, "2_code_only"), cfg, prompts, mode="code", title=title + " (code only)", eps=eps)
        out = {"summary_ai": res["summary_ai"]}
        if golden:
            out["golden"] = ev.golden_score_pipeline(golden, res)
            ev.write_json(os.path.join(out_root, "2_code_only", "golden_score.json"), out["golden"])
            summary["code_only"] = {"golden_recall": out["golden"]["recall"], "flags": res["summary_ai"]}
        return out

    def planted(tag, allowed, per_type, seed):
        d = os.path.join(out_root, f"3_planted_{tag}" if tag == "test" else f"5_planted_{tag}")
        avail = sorted(e.num for e in eps)
        allowed_eff = (allowed & set(avail)) or {n for i, n in enumerate(avail) if i % 2 == (0 if tag == "test" else 1)}
        m_eps, muts = ev.plant_errors(eps, seed=seed, allowed=allowed_eff, per_type=per_type)
        ev.write_json(os.path.join(d, "answer_key.json"), muts)
        res_full = run_check(None, os.path.join(d, "full"), cfg, prompts, mode=mode, llm=llm, title=f"planted {tag} (full)", eps=m_eps)
        res_code = run_check(None, os.path.join(d, "code_only"), cfg, prompts, mode="code", title=f"planted {tag} (code only)", eps=m_eps)
        s_full = ev.score(muts, lambda m: ev.caught_by_pipeline(m, res_full))
        s_code = ev.score(muts, lambda m: ev.caught_by_pipeline(m, res_code))
        ev.write_json(os.path.join(d, "scores.json"), {"full": s_full, "code_only": s_code})
        summary[f"planted_{tag}"] = {"planted": len(muts), "full": _short(s_full), "code_only": _short(s_code)}
        return {"eps": m_eps, "muts": muts}

    def baseline(test_set):
        d = os.path.join(out_root, "4_baseline_one_prompt")
        b_show = ev.run_baseline_and_save(eps, llm, prompts, cfg, os.path.join(d, "on_show"))
        out = {}
        if golden:
            out["golden"] = ev.golden_score_baseline(golden, b_show["issues"])
            ev.write_json(os.path.join(d, "on_show", "golden_score.json"), out["golden"])
        if test_set:
            b_test = ev.run_baseline_and_save(test_set["eps"], llm, prompts, cfg, os.path.join(d, "on_planted_test"))
            s = ev.score(test_set["muts"], lambda m: ev.caught_by_baseline(m, b_test["issues"]))
            ev.write_json(os.path.join(d, "on_planted_test", "scores.json"), s)
            out["planted_test"] = _short(s)
        summary["baseline"] = {"golden_recall": out.get("golden", {}).get("recall"), "planted_test": out.get("planted_test"),
                               "issues_on_show": len(b_show["issues"])}
        return out

    def clean():
        d = os.path.join(out_root, "6_clean_control")
        r_full = run_check(clean_show, os.path.join(d, "full"), cfg, prompts, mode=mode, llm=llm, title="clean control (full)")
        r_code = run_check(clean_show, os.path.join(d, "code_only"), cfg, prompts, mode="code", title="clean control (code only)")
        summary["clean_control"] = {"full": r_full["summary_ai"], "code_only": r_code["summary_ai"]}
        return summary["clean_control"]

    safe("main", main)
    safe("code", code)
    test_set = safe("test", lambda: planted("test", TEST_EPS, per_type_test, seed=29))
    if llm is None:
        summary["steps"]["baseline"] = "skipped (no model in code-only mode)"
    else:
        safe("baseline", lambda: baseline(test_set))
    safe("dev", lambda: planted("dev", DEV_EPS, per_type_dev, seed=11))
    if clean_show:
        safe("clean", clean)
    usage = getattr(llm, "usage", None)
    summary["usage"] = usage.__dict__ if usage else None
    _save(out_root, summary)
    _write_md(out_root, summary)
    return summary


def _short(s: dict) -> dict:
    return {"in_scope_recall": s["in_scope_recall"], "caught": s["in_scope_caught"], "planted_in_scope": s["in_scope_planted"],
            "per_type": s["per_type"]}


def _save(out_root, summary):
    with open(os.path.join(out_root, "00_evaluation_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)


def _write_md(out_root, s):
    L = ["# Evaluation summary", ""]
    L.append("| Step | Status |")
    L.append("|---|---|")
    for k, v in s["steps"].items():
        L.append(f"| {k} | {v}{' - ' + s['errors'][k] if k in s.get('errors', {}) else ''} |")
    if "main" in s and s["main"].get("golden"):
        g = s["main"]["golden"]
        L += ["", f"**Golden set (real issues in the show), full pipeline:** {g['hits']}/{g['items_total']} = {g['recall']:.0%}"]
    if "code_only" in s:
        L.append(f"**Golden set, code only:** {s['code_only']['golden_recall']:.0%}")
    if s.get("baseline", {}).get("golden_recall") is not None:
        L.append(f"**Golden set, one-prompt baseline:** {s['baseline']['golden_recall']:.0%}")
    for tag in ("planted_test", "planted_dev"):
        if tag in s:
            p = s[tag]
            if not p["planted"]:
                L += ["", f"## {tag.replace('_', ' ')}: no errors could be planted in this script"]
                continue
            L += ["", f"## {tag.replace('_', ' ')} ({p['planted']} planted errors)", "", "| Method | In-scope catch rate |", "|---|---|",
                  f"| Full pipeline (code + AI) | {p['full']['caught']}/{p['full']['planted_in_scope']} = {p['full']['in_scope_recall']:.0%} |",
                  f"| Code only | {p['code_only']['caught']}/{p['code_only']['planted_in_scope']} = {p['code_only']['in_scope_recall']:.0%} |"]
            if tag == "planted_test" and s.get("baseline", {}).get("planted_test"):
                b = s["baseline"]["planted_test"]
                L.append(f"| One-prompt AI baseline | {b['caught']}/{b['planted_in_scope']} = {b['in_scope_recall']:.0%} |")
            L += ["", "| Error type | Full | Code only |", "|---|---|---|"]
            for t, v in p["full"]["per_type"].items():
                c = p["code_only"]["per_type"].get(t, {"caught": 0, "planted": 0})
                L.append(f"| {t} | {v['caught']}/{v['planted']} | {c['caught']}/{c['planted']} |")
    if "clean_control" in s:
        c = s["clean_control"]
        L += ["", f"**Clean control (no errors planted):** full mode raised {c['full']['high']} High / {c['full']['flag']} flags; code only {c['code_only']['high']} High / {c['code_only']['flag']} flags."]
    if s.get("usage"):
        u = s["usage"]
        L += ["", f"**Model usage:** {u['calls']} calls ({u['cached_calls']} from cache), {u['prompt_tokens']:,} input + {u['completion_tokens']:,} output tokens, {u['seconds']:.0f}s"]
    with open(os.path.join(out_root, "00_evaluation_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
