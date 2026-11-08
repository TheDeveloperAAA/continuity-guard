"""ContinuityGuard command line.

  python run.py check    --input SHOW --out OUT [--mode full|code|mock] [--decisions FILE] [--model NAME]
  python run.py evaluate --input SHOW --out OUT [--golden FILE] [--decisions FILE] [--clean DIR] [--steps main,code,test,baseline,dev,clean]
  python run.py smoke    [--model NAME]          # one tiny call per model, to see which ones your key can use

SHOW can be a folder of episode .txt files, a .zip, or a single file.
"""
import argparse
import json
import os
import sys
import time

from cguard.llm import LLMError, build_llm
from cguard.pipeline import load_cfg, load_decisions, load_prompts, run_check
from cguard.suite import run_suite


def _llm(cfg, args, out_dir):
    if args.mode == "code":
        return None
    if args.mode == "mock":
        from cguard.mock import MockLLM
        return MockLLM(os.path.join(out_dir, "llm_calls"))
    return build_llm(cfg, out_dir, cache_dir=args.cache)


def main(argv=None):
    ap = argparse.ArgumentParser(description="ContinuityGuard: catch continuity drift in AI-written microdrama scripts")
    ap.add_argument("command", choices=["check", "evaluate", "smoke"])
    ap.add_argument("--input")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--mode", default="full", choices=["full", "code", "mock"])
    ap.add_argument("--decisions")
    ap.add_argument("--golden")
    ap.add_argument("--clean")
    ap.add_argument("--steps", default="main,code,test,baseline,dev,clean")
    ap.add_argument("--title", default="")
    ap.add_argument("--model", help="override the model name from config.yaml")
    ap.add_argument("--fallbacks", help="comma-separated backup models (overrides config.yaml)")
    ap.add_argument("--candidates", help="smoke: comma-separated models to try")
    ap.add_argument("--save", help="smoke: write the list of working models to this JSON file")
    ap.add_argument("--base-url", help="override the endpoint from config.yaml")
    ap.add_argument("--config")
    ap.add_argument("--prompts")
    ap.add_argument("--cache", default=".cg_cache")
    args = ap.parse_args(argv)

    cfg, prompts = load_cfg(args.config), load_prompts(args.prompts)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.fallbacks is not None:
        cfg["model"]["fallbacks"] = [m.strip() for m in args.fallbacks.split(",") if m.strip()]
    if args.base_url:
        cfg["model"]["base_url"] = args.base_url

    if args.command == "smoke":
        names = [m.strip() for m in (args.candidates or "").split(",") if m.strip()] or \
            [cfg["model"]["name"]] + cfg["model"].get("fallbacks", [])
        from concurrent.futures import ThreadPoolExecutor

        def probe(name):
            c = json.loads(json.dumps(cfg))
            c["model"]["name"], c["model"]["fallbacks"], c["model"]["max_retries"] = name, [], 2
            t = time.time()
            try:
                build_llm(c, None, cache_dir=None).chat_json("Reply in JSON.", 'Return {"ok": true}', stage="smoke")
                return name, True, f"{time.time() - t:4.1f}s"
            except LLMError as e:
                msg = str(e).replace("\n", " ")
                return name, False, ("busy (503)" if "HTTP 503" in msg else "not available (404)" if "HTTP 404" in msg else
                                     "quota/rate limit (429)" if "HTTP 429" in msg else msg[:90])

        working = []
        with ThreadPoolExecutor(max_workers=8) as pool:  # all models are checked at the same time
            for name, ok, info in pool.map(probe, list(dict.fromkeys(names))):
                print(f"{'OK  ' if ok else 'FAIL'}  {name:28s} {info}")
                if ok:
                    working.append(name)
        print("WORKING:", ", ".join(working) or "none")
        if args.save:
            with open(args.save, "w") as fh:
                json.dump({"working": working}, fh)
        return 0 if working else 1

    if not args.input:
        ap.error("--input is required")
    os.makedirs(args.out, exist_ok=True)
    title = args.title or os.path.basename(os.path.normpath(args.input))
    if args.command == "check":
        llm = _llm(cfg, args, args.out)
        mode = "code" if args.mode == "code" else "full"
        res = run_check(args.input, args.out, cfg, prompts, mode=mode, decisions=load_decisions(args.decisions), llm=llm, title=title)
        print(json.dumps({"before_human": res["summary_ai"], "after_human": res.get("summary_final")}, indent=1))
        print(f"Report: {os.path.join(args.out, '05_ai_report.html')}")
        return 0
    llm = _llm(cfg, args, args.out)
    golden = json.load(open(args.golden)) if args.golden else None
    steps = tuple(s.strip() for s in args.steps.split(",") if s.strip())
    s = run_suite(args.input, args.out, cfg, prompts, llm, golden=golden, decisions=load_decisions(args.decisions),
                  clean_show=args.clean, steps=steps, mode="code" if args.mode == "code" else "full", title=title)
    print(open(os.path.join(args.out, "00_evaluation_summary.md")).read())
    return 0 if not s["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
