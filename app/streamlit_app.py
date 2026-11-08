"""ContinuityGuard web app (Streamlit).

A script editor uploads a show (or uses the built-in demo), runs the check, and gets the issues to decide,
every flag with its exact quote and timestamp, the full report, a series bible and, with decisions, corrected scripts.
The AI steps run live with the visitor's own key, replay a recorded run, or are skipped (code only).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cguard  # noqa: E402
from cguard.llm import REPLAY_KEY, load_recorded_answers  # noqa: E402

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
RECORDED_MODEL = "gemini-3.6-flash"          # primary model of the recorded live run
REPO = "https://github.com/TheDeveloperAAA/continuity-guard"
COLAB = "https://colab.research.google.com/github/TheDeveloperAAA/continuity-guard/blob/main/ContinuityGuard.ipynb"
WORK_ROOT = Path(tempfile.gettempdir()) / "continuityguard"

st.set_page_config(page_title="ContinuityGuard", page_icon=":material/fact_check:", layout="wide")


# ---------------------------------------------------------------- helpers
def workdir() -> Path:
    """One private temporary folder per browser session; folders older than 2 hours are deleted."""
    if "wd" not in st.session_state:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        for old in WORK_ROOT.iterdir():
            try:
                if time.time() - old.stat().st_mtime > 2 * 3600:
                    shutil.rmtree(old, ignore_errors=True)
            except OSError:
                pass
        st.session_state.wd = Path(tempfile.mkdtemp(prefix="s_", dir=WORK_ROOT))
    return st.session_state.wd


def save_upload(files, dest: Path) -> Path | None:
    """Save uploaded episode files. One .zip is used as is; .txt files (or several zips) go into one folder."""
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    if len(files) == 1 and files[0].name.lower().endswith(".zip"):
        p = dest / Path(files[0].name).name
        p.write_bytes(files[0].getvalue())
        return p
    folder = dest / "episodes"
    folder.mkdir()
    for f in files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                z.extractall(folder / Path(f.name).stem)
        else:
            (folder / Path(f.name).name).write_bytes(f.getvalue())
    return folder


def run_cli(args: list[str], env: dict, box) -> tuple[int, str]:
    """Run run.py and show its log live."""
    p = subprocess.Popen([sys.executable, "run.py"] + args, cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-18:]), language=None)
    return p.wait(), "\n".join(lines)


def md_table(md: str, heading: str) -> pd.DataFrame:
    """Read the first markdown table under a heading of the tool's own report."""
    if heading not in md:
        return pd.DataFrame()
    rows = []
    for line in md.split(heading, 1)[1].splitlines()[1:]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    if len(rows) < 2:
        return pd.DataFrame()
    return pd.DataFrame(rows[2:], columns=rows[0])


# ---------------------------------------------------------------- header
st.title("ContinuityGuard")
st.markdown("**Catch continuity drift in AI-written microdrama scripts before any voice or video is generated:** "
            "the same name spelled differently, one character under several speaker labels, and contradicting "
            "appearance (eye or hair colour, scars, signature items). Every flag gives the episode, timestamp, "
            "an exact quote cut from the script by code, a severity and a ready-made fix.")
with st.expander("How it works, and how well it works"):
    st.markdown(f"""
**Pipeline:** parse every block with its timestamp → **code checks** (near-identical spellings, speaker labels,
colour, scar and item words; severity: spoken line = High, camera/action = Medium, notes = Low) → **AI Reader**
(10 episodes per request: what each name refers to, who each speaker label is) → **AI Judge** (same or different?
mistake or deliberate?) → **code guards** → the editor decides → corrected scripts, a series bible and a continuity
prompt block for the next episode.

**Guards:** every quote is cut from the script by code, never written by the model; citations that do not match the
script are rejected; the AI may add or explain findings but cannot delete what code found (disagreements go to
*needs review*); names used together in one line are never merged.

**On a real 50-episode Kuku show (the assignment pack), final run, Gemini 3.6 Flash:**

| Test | Code + AI | Code only | One AI prompt |
|---|---|---|---|
| Real issues, verified by hand (21) | 18/21 = 86% | 57% | 52% |
| Frozen test set of planted errors (19) | 18/19 = 95% | 84% | 84% |
| False alarms on a clean script | 0 | 0 | - |

Cost per 50-episode show: 7 model requests, about $0.48 at Gemini 3.6 Flash list prices. Editor work: about 16 decisions.
Code: [GitHub]({REPO}) · Notebook: [Colab]({COLAB})
""")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · Script")
    source = st.radio("Show to check", ["Demo show (3 episodes, built in)", "Upload a show"], label_visibility="collapsed")
    uploads = []
    if source == "Upload a show":
        uploads = st.file_uploader("Episodes: one .zip, or several .txt files", type=["zip", "txt"],
                                   accept_multiple_files=True)
        st.caption("Files are processed in a temporary folder that is deleted within 2 hours. Nothing is published.")

    st.header("2 · AI")
    mode = st.radio("AI steps", ["Code only (free, instant)", "Replay a recorded run", "Live AI (your API key)"],
                    label_visibility="collapsed")
    recorded = key = None
    model, base_url = RECORDED_MODEL, GEMINI_URL
    if mode.startswith("Replay"):
        recorded = st.file_uploader("raw_outputs.zip from an earlier run", type=["zip"])
        st.caption("Reuses the model's recorded answers: no key, no cost. It reproduces a run only on the same show.")

    # a form sends the typed key together with the click, so the key never needs Enter first
    with st.form("run_form", border=False):
        if mode.startswith("Live"):
            key = st.text_input("API key", type="password", help="Used for this run only; never stored or logged.")
            model = st.text_input("Model", value=RECORDED_MODEL)
            base_url = st.text_input("Endpoint (any OpenAI-compatible API)", value=GEMINI_URL)
            st.caption("A 50-episode show takes about 7 minutes on Gemini's free tier.")

        st.header("3 · Editor decisions")
        decisions = None
        if source.startswith("Demo"):
            if st.checkbox("Apply the example decisions", value=True,
                           help="Shows the last step: corrected scripts and a diff."):
                decisions = ROOT / "decisions.example.yaml"
        else:
            dec_up = st.file_uploader("decisions.yaml (optional): corrected scripts + diff", type=["yaml", "yml"])
            if dec_up:
                decisions = workdir() / "decisions.yaml"
                decisions.write_bytes(dec_up.getvalue())

        go = st.form_submit_button("Run ContinuityGuard", type="primary", use_container_width=True)
    st.caption(f"v{cguard.__version__} · [Code]({REPO}) · [Colab]({COLAB})")

# ---------------------------------------------------------------- run
if go:
    wd = workdir()
    problem = None
    if source.startswith("Demo"):
        show = ROOT / "samples" / "demo_show"
    elif not uploads:
        problem = "Upload a .zip of episode .txt files, or several .txt files, first."
    else:
        show = save_upload(uploads, wd / "input")
    if mode.startswith("Replay") and not recorded:
        problem = problem or "Upload the raw_outputs.zip of the run you want to replay."
    if mode.startswith("Live") and not (key or "").strip():
        problem = problem or "Paste an API key, or choose Code only."
    if problem:
        st.warning(problem)
    else:
        out, cache = wd / "outputs" / "check", wd / "cache"
        shutil.rmtree(out, ignore_errors=True)
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        args = ["check", "--input", str(show), "--out", str(out), "--cache", str(cache)]
        t0 = time.time()
        with st.status("Running ContinuityGuard…", expanded=True) as status:
            box = st.empty()
            if mode.startswith("Code"):
                args += ["--mode", "code"]
                how = "code only"
            elif mode.startswith("Replay"):
                rec = wd / "raw_outputs.zip"
                rec.write_bytes(recorded.getvalue())
                n = load_recorded_answers(str(rec), str(cache), RECORDED_MODEL)
                env["LLM_API_KEY"] = REPLAY_KEY
                args += ["--mode", "full", "--model", RECORDED_MODEL, "--fallbacks", "", "--base-url", GEMINI_URL]
                how = f"replay of {n} recorded answers"
            else:
                env["LLM_API_KEY"] = key.strip()
                cands = [model] + (["gemini-flash-latest", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-lite-latest"]
                                   if "generativelanguage" in base_url else [])
                saved = wd / "working_models.json"
                st.write("Checking which models this key can reach…")
                run_cli(["smoke", "--candidates", ",".join(dict.fromkeys(cands)), "--base-url", base_url,
                         "--save", str(saved)], env, box)
                working = (pd.read_json(saved)["working"].tolist() if saved.exists() and saved.stat().st_size > 15 else [])
                if working:
                    args += ["--mode", "full", "--model", working[0], "--fallbacks", ",".join(working[1:4]), "--base-url", base_url]
                    how = f"live AI with {working[0]}"
                else:
                    args += ["--mode", "code"]
                    how = "code only (no model answered with this key)"
            if decisions:
                args += ["--decisions", str(decisions)]
            st.write(f"Mode: {how}")
            rc, log = run_cli(args, env, box)
            secs = time.time() - t0
            status.update(label=f"{'Done' if rc == 0 else 'Failed'} in {secs:.0f}s · {how}",
                          state="complete" if rc == 0 else "error", expanded=rc != 0)
        st.session_state.result = {"out": out, "log": log, "rc": rc, "how": how}

# ---------------------------------------------------------------- results
res = st.session_state.get("result")
if not res:
    st.info("Choose a script and a mode in the sidebar, then press **Run ContinuityGuard**. "
            "The built-in demo runs in a few seconds with no key.")
    st.stop()
if res["rc"] != 0 or not res["out"].exists():
    st.error("The run did not finish. The end of the log:")
    st.code("\n".join(res["log"].splitlines()[-30:]), language=None)
    st.stop()

out = res["out"]
final = "06" if (out / "06_final_flags.csv").exists() else "05"
report_md = (out / f"{final}_{'final' if final == '06' else 'ai'}_report.md").read_text(encoding="utf-8")
flags = pd.read_csv(out / f"{final}_{'final' if final == '06' else 'ai'}_flags.csv").fillna("")
issues = md_table(report_md, "## Issues to decide")
summary = report_md.split("## Summary", 1)[1].split("## ", 1)[0] if "## Summary" in report_md else ""

st.subheader("Result" + (" (after editor decisions)" if final == "06" else ""))
c1, c2, c3, c4 = st.columns(4)
to_decide = issues[issues.get("Status", pd.Series(dtype=str)).isin(["flag", "needs_review"])] if not issues.empty else issues
c1.metric("Issues to decide", len(to_decide))
c2.metric("High flags (spoken lines)", int(((flags.status == "flag") & (flags.severity == "High")).sum()))
c3.metric("Groups needing review", int(issues["Status"].eq("needs_review").sum()) if not issues.empty else 0)
c4.metric("Flagged mentions", int(flags.status.isin(["flag", "needs_review"]).sum()))
st.markdown(summary)

tabs = st.tabs(["Issues to decide", "All flags", "Full report", "Series bible", "Corrected scripts", "Downloads & log"])
with tabs[0]:
    st.caption("One decision fixes every mention in its group. Status 'info' = deliberate aliases to map, not errors.")
    st.dataframe(issues, use_container_width=True, hide_index=True)
with tabs[1]:
    sev = st.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    stat = st.multiselect("Status", sorted(flags.status.unique()), default=[s for s in ["flag", "needs_review"] if s in set(flags.status)])
    view = flags[flags.severity.isin(sev) & flags.status.isin(stat)]
    cols = [c for c in ["flag_id", "severity", "status", "episode", "where", "kind", "speaker", "found", "canonical",
                        "snippet", "fix", "source", "reason"] if c in view.columns]
    st.dataframe(view[cols], use_container_width=True, hide_index=True)
    st.download_button("Download these flags (CSV)", view[cols].to_csv(index=False).encode(), "flags.csv", "text/csv")
with tabs[2]:
    html = (out / f"{final}_{'final' if final == '06' else 'ai'}_report.html").read_text(encoding="utf-8")
    components.html(html, height=900, scrolling=True)
with tabs[3]:
    bible = out / "07_series_bible.md"
    block = out / "07_prompt_block.txt"
    if block.exists():
        st.markdown("**Continuity block for the next episode's writing prompt**")
        st.code(block.read_text(encoding="utf-8"), language=None)
    if bible.exists():
        st.markdown(bible.read_text(encoding="utf-8"))
with tabs[4]:
    diff = out / "08_fixes.diff"
    if diff.exists():
        lines = diff.read_text(encoding="utf-8").splitlines()
        changed = sum(l.startswith("+") and not l.startswith("+++") for l in lines)
        n_files = len(list((out / "08_fixed_scripts").glob("*")))
        st.markdown(f"**{changed} lines changed across {n_files} episodes.** First 300 lines of the diff:")
        st.code("\n".join(lines[:300]), language="diff")
    else:
        st.info("Corrected scripts are written when editor decisions are given (step 3 in the sidebar).")
with tabs[5]:
    wd = workdir()
    bundle = shutil.make_archive(str(wd / "continuityguard_outputs"), "zip", out)
    st.download_button("Download every output (zip)", Path(bundle).read_bytes(), "continuityguard_outputs.zip",
                       "application/zip", type="primary")
    st.caption(f"Mode: {res['how']}")
    st.code(res["log"][-6000:], language=None)
