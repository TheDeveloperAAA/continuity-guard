"""Basic tests. Run: python -m pytest -q   (or: python tests/test_basic.py)"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cguard import ai, evaluate as ev, issues  # noqa: E402
from cguard.mock import MockLLM  # noqa: E402
from cguard.parser import block_index, load_show, parse_episode  # noqa: E402
from cguard.pipeline import load_cfg, load_prompts, run_check  # noqa: E402

DEMO = os.path.join(ROOT, "samples", "demo_show")
CLEAN = os.path.join(ROOT, "samples", "demo_show_clean")
CFG, PROMPTS = load_cfg(), load_prompts()


def test_parse_timed():
    eps = load_show(DEMO)
    assert [e.num for e in eps] == [1, 2, 3]
    assert all(e.timed and len(e.blocks) >= 10 for e in eps)
    b = eps[0].blocks[2]
    assert b.kind == "DIALOGUE" and b.speaker == "VIKRAM" and b.where == "00:06.70"


def test_parse_fallback_untimed_with_hindi():
    txt = "INT. CHAI STALL - DAY\n\nMeera pours tea.\n\nMEERA (softly)\nचाय ठंडी हो गई, Arjun.\n\nARJUN\nShort hair suits you."
    ep = parse_episode(txt, "ep7_draft.txt")
    assert ep.num == 7 and not ep.timed
    kinds = [b.kind for b in ep.blocks]
    assert kinds == ["SCENE", "ACTION", "DIALOGUE", "DIALOGUE"]
    assert ep.blocks[2].speaker == "MEERA" and "चाय" in ep.blocks[2].text and ep.blocks[2].where == "line 5"


def test_code_mode_finds_planted_and_is_quiet_on_clean():
    with tempfile.TemporaryDirectory() as d:
        r = run_check(DEMO, os.path.join(d, "a"), CFG, PROMPTS, mode="code")
        multi = [set(g["members"]) for g in r["groups"] if len(g["members"]) > 1]
        assert {"Vikram", "Vikrant"} in multi
        assert any({"Raichand", "Raichandani", "Raychand"} <= m for m in multi)
        assert any({"MEERA", "MS. MEERA"} <= m for m in multi)
        assert any(t["trait"] == "eye_colour" for t in r["trait_groups"])
        r2 = run_check(CLEAN, os.path.join(d, "b"), CFG, PROMPTS, mode="code")
        assert r2["summary_ai"]["flag"] == 0


def test_guard_rejects_invented_citations():
    eps = load_show(DEMO)
    out = {"names": [], "speakers": [], "traits": [], "rejected": [], "quote_mismatches": 0}
    fake = {"names": [{"name": "Raichand", "kind": "organisation", "refers_to": "x", "block": "E01-B999", "quote": "x"},
                      {"name": "Zorawar", "kind": "person", "refers_to": "x", "block": "E01-B001", "quote": "x"},
                      {"name": "Meera", "kind": "person", "refers_to": "heroine", "block": "E01-B002", "quote": "Meera steps onto the stage"}],
            "speakers": [{"label": "NOBODY", "identity": "x"}], "traits": []}
    ai._validate_reader(fake, block_index(eps), ["MEERA", "VIKRAM"], out)
    assert len(out["names"]) == 1 and out["names"][0]["name"] == "Meera"
    assert len(out["rejected"]) == 3


def test_full_pipeline_with_mock_and_human_fixes():
    with tempfile.TemporaryDirectory() as d:
        dec = {"names": [{"canonical": "Raichand", "same_as": ["Raichandani", "Raychand"]}],
               "speakers": [{"canonical": "MEERA", "same_person": ["MS. MEERA"], "deliberate": False}]}
        r = run_check(DEMO, d, CFG, PROMPTS, mode="full", llm=MockLLM(os.path.join(d, "llm_calls")), decisions=dec)
        for f in ("00_run_manifest.json", "05_ai_report.html", "06_final_flags.csv", "07_series_bible.md", "08_fixes.diff"):
            assert os.path.exists(os.path.join(d, f)), f
        diff = open(os.path.join(d, "08_fixes.diff"), encoding="utf-8").read()
        assert "+\"The Raichand Group cancelled" in diff and "+**MEERA (Breathless):**" in diff


def test_planted_errors_and_scoring():
    eps = load_show(CLEAN)
    m_eps, muts = ev.plant_errors(eps, seed=3, allowed={1, 2, 3}, per_type=1)
    assert muts and all(m["planted"] != m["original"] for m in muts)
    assert len({m["block"] for m in muts}) == len(muts)
    with tempfile.TemporaryDirectory() as d:
        r = run_check(None, d, CFG, PROMPTS, mode="code", eps=m_eps)
        s = ev.score(muts, lambda m: ev.caught_by_pipeline(m, r))
        assert 0 <= s["in_scope_recall"] <= 1 and sum(v["planted"] for v in s["per_type"].values()) == len(muts)



def test_llm_client_against_local_openai_compatible_server():
    """The real client: drops JSON mode if refused, retries on 429, strips code fences, caches replies."""
    import http.server
    import json
    import threading
    from cguard.llm import LLM
    calls = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            if len(calls) == 1:
                code, payload = 400, {"error": {"message": "response_format json_object is not supported"}}
            elif len(calls) == 2:
                code, payload = 429, {"error": {"message": "rate limited"}}
            else:
                code, payload = 200, {"choices": [{"message": {"content": "```json\n{\"ok\": true,}\n```"}}],
                                      "usage": {"prompt_tokens": 11, "completion_tokens": 3}}
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            if code == 429:
                self.send_header("Retry-After", "0.1")
            self.end_headers()
            self.wfile.write(raw)

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as d:
        llm = LLM(f"http://127.0.0.1:{srv.server_port}/v1", "test-model", "k", cache_dir=d, max_retries=2, max_output_tokens=100)
        assert llm.chat_json("sys", "user", stage="t") == {"ok": True}
        assert "response_format" in calls[0] and "response_format" not in calls[-1] and calls[-1]["max_tokens"] == 100
        n = len(calls)
        assert llm.chat_json("sys", "user", stage="t") == {"ok": True} and len(calls) == n  # second time from cache
        assert llm.usage.cached_calls == 1 and llm.usage.prompt_tokens == 11
    srv.shutdown()



def test_rate_limits_wait_instead_of_burning_models():
    """429 per-minute: pause and retry the same model. 429 per-day: move to the next model at once."""
    import http.server
    import json
    import threading
    from cguard.llm import LLM
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body["model"])
            if body["model"] == "daily-capped":
                code, payload = 429, [{"error": {"code": 429, "message": "quota", "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}]
            elif body["model"] == "minute-capped" and seen.count("minute-capped") <= 2:
                code, payload = 429, [{"error": {"code": 429, "message": "Please retry in 0.2s.", "details": [{"retryDelay": "0s"}]}}]
            else:
                code, payload = 200, {"choices": [{"message": {"content": '{"model": "%s"}' % body["model"]}}]}
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}/v1"
    a = LLM(base, "minute-capped", "k", fallbacks=["backup"], json_mode=False)
    assert a.chat_json("s", "u1", stage="t") == {"model": "minute-capped"}  # waited, did not switch
    assert "backup" not in seen
    b = LLM(base, "daily-capped", "k", fallbacks=["backup"], json_mode=False)
    assert b.chat_json("s", "u2", stage="t") == {"model": "backup"}  # switched straight away
    assert seen.count("daily-capped") == 1
    srv.shutdown()

if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)} tests passed")
