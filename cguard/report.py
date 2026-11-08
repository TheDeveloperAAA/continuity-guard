"""Write the editor-facing outputs: Markdown, CSV and HTML reports, plus the series bible."""
from __future__ import annotations

import csv
import html
import json
import os
from collections import Counter, defaultdict

FLAG_COLS = ["flag_id", "group_id", "type", "severity", "status", "episode", "block", "where", "kind", "speaker",
             "found", "canonical", "snippet", "fix", "verdict", "confidence", "source", "reason"]
SEV_ORDER = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}


def _group_rows(groups, tgroups, flags):
    by_group = defaultdict(list)
    for f in flags:
        by_group[f["group_id"]].append(f)
    rows = []
    for g in groups:
        fl = by_group.get(g["group_id"], [])
        if not fl:
            continue
        variants = [m for m in g["members"] if m != g["canonical"]]
        eps = sorted({f["episode"] for f in fl})
        rows.append({"group_id": g["group_id"], "type": g["type"], "canonical": g["canonical"], "variants": variants,
                     "episodes": eps, "mentions": len(fl), "spoken": sum(f["kind"] == "DIALOGUE" for f in fl),
                     "verdict": g["verdict"], "confidence": g["confidence"], "source": g["source"], "reason": g["reason"],
                     "status": "info" if all(f["status"] == "info" for f in fl) else ("flag" if any(f["status"] == "flag" for f in fl) else "needs_review"),
                     "top_sev": min((SEV_ORDER[f["severity"]] for f in fl), default=3)})
    for t in tgroups:
        gid = t.get("group_id") or t.get("cand")
        fl = by_group.get(gid, [])
        if not fl:
            continue
        rows.append({"group_id": gid, "type": "TRAIT", "canonical": f'{t["character"]}: {t["canonical"]}',
                     "variants": t["values_to_fix"], "episodes": sorted({f["episode"] for f in fl}), "mentions": len(fl),
                     "spoken": sum(f["kind"] == "DIALOGUE" for f in fl), "verdict": t["verdict"], "confidence": t["confidence"],
                     "source": t["source"], "reason": t["reason"],
                     "status": "flag" if any(f["status"] == "flag" for f in fl) else "needs_review",
                     "top_sev": min((SEV_ORDER[f["severity"]] for f in fl), default=3)})
    rows.sort(key=lambda r: ({"flag": 0, "needs_review": 1, "info": 2}[r["status"]], r["top_sev"], -r["mentions"]))
    return rows


def _cleared(groups, tgroups):
    out = []
    for g in groups:
        if g["verdict"] in {"distinct", "deliberate"} and len(g["members"]) >= 1 and g["source"] != "code":
            if g["verdict"] == "distinct" and len(g["members"]) == 1 and g["source"] != "human":
                continue
            out.append(f'{g["type"]} {", ".join(g["members"])}: {g["verdict"]} ({g["source"]}) - {g["reason"]}')
    for t in tgroups:
        if t["verdict"] in {"deliberate", "compatible"}:
            out.append(f'TRAIT {t["character"]} {t["trait"]}: {t["verdict"]} - {t["reason"]}')
    return out


def write_reports(out_dir: str, prefix: str, title: str, meta: dict, groups, tgroups, flags) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    for i, t in enumerate(tgroups):
        t.setdefault("group_id", t.get("cand", f"T{i + 1:02d}"))
    flags = sorted(flags, key=lambda f: ({"flag": 0, "needs_review": 1, "info": 2}[f["status"]], SEV_ORDER[f["severity"]], f["episode"], f["block"]))
    rows = _group_rows(groups, tgroups, flags)
    sev = Counter(f["severity"] for f in flags if f["status"] == "flag")
    st = Counter(f["status"] for f in flags)
    summary = {
        "flags_total": len(flags), "flag": st.get("flag", 0), "needs_review": st.get("needs_review", 0), "info": st.get("info", 0),
        "high": sev.get("High", 0), "medium": sev.get("Medium", 0), "low": sev.get("Low", 0),
        "issue_groups": len([r for r in rows if r["status"] != "info"]),
        "spoken_mentions": sum(1 for f in flags if f["kind"] == "DIALOGUE" and f["status"] != "info"),
    }
    # CSV
    csv_path = os.path.join(out_dir, f"{prefix}_flags.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FLAG_COLS, extrasaction="ignore")
        w.writeheader()
        for f in flags:
            w.writerow(f)
    # Markdown
    md = [f"# ContinuityGuard report: {title}", "",
          f"Model: {meta.get('model', '-')} | Mode: {meta.get('mode')} | Episodes: {meta.get('episodes')} | Blocks: {meta.get('blocks')} | Run: {meta.get('run_at')}", "",
          "## Summary", "",
          f"- **{summary['issue_groups']} issue groups to decide** (one decision fixes every mention in its group)",
          f"- Flags: **{summary['high']} High** (spoken), {summary['medium']} Medium (camera/action/labels), {summary['low']} Low (notes) | "
          f"{summary['needs_review']} need review | {summary['info']} info rows (deliberate aliases to map)",
          f"- Mentions inside spoken dialogue: {summary['spoken_mentions']}"]
    if meta.get("reader_rejected") is not None:
        md.append(f"- Model citations rejected by code checks: {meta['reader_rejected']} of {meta['reader_items']} (never used)")
    md += ["", "## Issues to decide", "", "| Group | Type | Use | Instead of | Episodes | Mentions | Spoken | Verdict | Conf | Source | Status |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['group_id']} | {r['type']} | {r['canonical']} | {', '.join(r['variants'])} | {_eps(r['episodes'])} | {r['mentions']} | {r['spoken']} | {r['verdict']} | {r['confidence']} | {r['source']} | {r['status']} |")
    cleared = _cleared(groups, tgroups)
    if cleared:
        md += ["", "## Checked and cleared (not errors)", ""] + [f"- {c}" for c in cleared[:40]]
    md += ["", "## Details (first 15 mentions per group; the CSV has all)", ""]
    by_group = defaultdict(list)
    for f in flags:
        by_group[f["group_id"]].append(f)
    for r in rows:
        md += [f"### {r['group_id']} {r['type']}: use \"{r['canonical']}\"", "", f"Reason: {r['reason']}", "",
               "| Flag | Sev | Ep | Where | Kind | Found | Snippet | Fix |", "|---|---|---|---|---|---|---|---|"]
        for f in by_group[r["group_id"]][:15]:
            md.append(f"| {f['flag_id']} | {f['severity']} | {f['episode']} | {f['where']} | {f['kind']} | {f['found']} | {f['snippet'].replace('|', '/')} | {f['fix']} |")
        md.append("")
    md_path = os.path.join(out_dir, f"{prefix}_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    html_path = os.path.join(out_dir, f"{prefix}_report.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(_html(title, meta, summary, rows, by_group, cleared))
    return {"summary": summary, "rows": rows, "paths": [csv_path, md_path, html_path]}


def _eps(eps):
    from .issues import ep_ranges
    return ep_ranges(eps)


def _html(title, meta, summary, rows, by_group, cleared) -> str:
    e = html.escape
    css = """body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#1b1b1b;max-width:1180px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:26px 0 8px}h3{font-size:14px;margin:18px 0 6px}
.meta{color:#666;font-size:12px}.kpis{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}
.kpi{border:1px solid #ddd;border-radius:8px;padding:10px 14px;min-width:120px}.kpi b{font-size:20px;display:block}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{border-bottom:1px solid #eee;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f6f6f6}.High{color:#b00020;font-weight:600}.Medium{color:#a15c00}.Low{color:#555}.Info{color:#2a6fdb}
.badge{border-radius:10px;padding:1px 8px;font-size:11px;background:#eee}.flag{background:#ffe5e5}.needs_review{background:#fff3cd}.info{background:#e5efff}
details{margin:6px 0}summary{cursor:pointer}"""
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>{e(title)}</title><style>{css}</style></head><body>",
           f"<h1>ContinuityGuard report: {e(title)}</h1>",
           f"<div class='meta'>Model: {e(str(meta.get('model', '-')))} | Mode: {e(str(meta.get('mode')))} | Episodes: {meta.get('episodes')} | Blocks: {meta.get('blocks')} | Run: {e(str(meta.get('run_at')))}</div>",
           "<div class='kpis'>"]
    for k, lab in (("issue_groups", "issue groups to decide"), ("high", "High (spoken)"), ("medium", "Medium"), ("low", "Low"),
                   ("needs_review", "need review"), ("info", "aliases to map")):
        out.append(f"<div class='kpi'><b>{summary[k]}</b>{lab}</div>")
    out.append("</div><h2>Issues to decide</h2><table><tr><th>Group</th><th>Type</th><th>Use</th><th>Instead of</th><th>Episodes</th><th>Mentions</th><th>Spoken</th><th>Verdict</th><th>Conf</th><th>Source</th><th>Status</th></tr>")
    for r in rows:
        out.append(f"<tr><td>{r['group_id']}</td><td>{r['type']}</td><td><b>{e(r['canonical'])}</b></td><td>{e(', '.join(r['variants']))}</td>"
                   f"<td>{_eps(r['episodes'])}</td><td>{r['mentions']}</td><td>{r['spoken']}</td><td>{r['verdict']}</td><td>{r['confidence']}</td>"
                   f"<td>{r['source']}</td><td><span class='badge {r['status']}'>{r['status']}</span></td></tr>")
    out.append("</table>")
    if cleared:
        out.append("<h2>Checked and cleared (not errors)</h2><ul>" + "".join(f"<li>{e(c)}</li>" for c in cleared[:40]) + "</ul>")
    out.append("<h2>Details</h2>")
    for r in rows:
        fl = by_group[r["group_id"]]
        out.append(f"<details><summary><b>{r['group_id']}</b> {r['type']}: use <b>{e(r['canonical'])}</b> ({len(fl)} mention{'s' if len(fl) != 1 else ''}) - {e(r['reason'])}</summary>"
                   "<table><tr><th>Flag</th><th>Sev</th><th>Ep</th><th>Where</th><th>Kind</th><th>Found</th><th>Snippet</th><th>Fix</th></tr>")
        for f in fl[:60]:
            out.append(f"<tr><td>{f['flag_id']}</td><td class='{f['severity']}'>{f['severity']}</td><td>{f['episode']}</td><td>{e(str(f['where']))}</td>"
                       f"<td>{f['kind']}</td><td>{e(f['found'])}</td><td>{e(f['snippet'])}</td><td>{e(f['fix'])}</td></tr>")
        out.append("</table></details>")
    out.append("</body></html>")
    return "\n".join(out)


def write_bible(out_dir: str, prefix: str, bible: dict, prompt_block: str) -> list[str]:
    jp = os.path.join(out_dir, f"{prefix}_series_bible.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(bible, fh, indent=2, ensure_ascii=False)
    md = ["# Series bible (continuity reference)", "", "## Names to use", ""]
    for n in bible["names"]:
        md.append(f"- **{n['canonical']}**" + (f" (never: {', '.join(n['do_not_use'])})" if n["do_not_use"] else f" ({n.get('note', '')})"))
    md += ["", "## Characters", "", "| Character | Speaker labels | Lines | Episodes | Traits |", "|---|---|---|---|---|"]
    for c in bible["characters"]:
        labs = "; ".join(f"{s['label']} ({s['lines']}{', ' + s['note'] if s['note'] else ''})" for s in c["speaker_labels"])
        tr = "; ".join(f"{k}: {v}" for k, v in c["traits"].items())
        md.append(f"| {c['name']} | {labs} | {c['lines']} | {c['episodes']} | {tr} |")
    mp = os.path.join(out_dir, f"{prefix}_series_bible.md")
    with open(mp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    pp = os.path.join(out_dir, f"{prefix}_prompt_block.txt")
    with open(pp, "w", encoding="utf-8") as fh:
        fh.write(prompt_block + "\n")
    return [jp, mp, pp]
