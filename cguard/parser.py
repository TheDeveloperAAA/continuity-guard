"""Parse microdrama screenplays into episodes and blocks.

Pure code, no AI. Two formats are supported:
1. Kuku-style timed scripts: **HEAD:** / Start: (mm:ss.s) | End: (mm:ss.s) / text
2. Fallback for anything else (e.g. plain screenplay text): paragraphs become blocks,
   ALL-CAPS cue lines become speakers, and locations are line numbers instead of times.
"""
from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass, field

TS = r"(\d+):(\d+(?:\.\d+)?)"
BLOCK_RE = re.compile(
    r"^\*\*(?P<head>.+?):\*\*[ \t]*\n\s*Start:\s*\(" + TS + r"\)\s*\|\s*End:\s*\(" + TS
    + r"\)[ \t]*\n(?P<body>.*?)(?=\n\s*\n|\n\*\*|\nSCENE\s+\d+|\n--- SCRIPT END|\Z)",
    re.M | re.S,
)
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.M)
HEADER_RE = {
    "number": re.compile(r"^EPISODE NUMBER:\s*(.+)$", re.M),
    "logline": re.compile(r"^EPISODE LOGLINE:\s*(.+)$", re.M),
    "continuity": re.compile(r"^CONTINUITY NOTES:\s*(.+)$", re.M),
}
VOICE_MODS = {"O.S.", "V.O.", "CONT'D", "OS", "VO", "O.C."}
WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


@dataclass
class Block:
    id: str
    ep: int
    scene: int
    kind: str          # CAMERA | ACTION | SOUND | DIALOGUE | SCENE | LOGLINE | NOTES
    speaker: str       # dialogue label as written (upper case), '' otherwise
    emotion: str
    start: float | None
    end: float | None
    text: str
    line: int = 0      # line number in the source file (always available)

    @property
    def words(self) -> int:
        return len(WORD_RE.findall(self.text))

    @property
    def where(self) -> str:
        """Human location: timestamp when the script is timed, else line number."""
        if self.start is None:
            return f"line {self.line}"
        return fmt_ts(self.start)


@dataclass
class Episode:
    num: int
    source: str
    logline: str = ""
    continuity: str = ""
    blocks: list[Block] = field(default_factory=list)
    timed: bool = True
    raw: str = ""

    @property
    def duration(self) -> float:
        return max((b.end or 0.0 for b in self.blocks), default=0.0)

    def all_units(self) -> list[Block]:
        """Script blocks plus the logline and continuity notes as addressable pseudo-blocks."""
        extra = []
        if self.logline:
            extra.append(Block(f"E{self.num:02d}-LOGLINE", self.num, 0, "LOGLINE", "", "", None, None, self.logline, 1))
        if self.continuity:
            extra.append(Block(f"E{self.num:02d}-NOTES", self.num, 0, "NOTES", "", "", None, None, self.continuity, 1))
        return extra + self.blocks


def fmt_ts(sec: float | None) -> str:
    if sec is None:
        return "-"
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def _secs(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def _classify(head: str) -> tuple[str, str, str]:
    h = head.strip()
    up = h.upper()
    if up.startswith("SOUND") or up in {"SFX", "MUSIC"}:
        return "SOUND", "", ""
    if up in {"CAMERA", "ACTION"}:
        return up, "", ""
    paren = re.findall(r"\(([^)]*)\)", h)
    name = re.sub(r"\s*\(.*$", "", h).strip()
    emo = ", ".join(p for p in paren if p.strip().upper() not in VOICE_MODS)
    return "DIALOGUE", name.upper(), emo


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _parse_timed(text: str, ep: Episode) -> None:
    scenes = [(mm.start(), int(mm.group(1))) for mm in SCENE_RE.finditer(text)]

    def scene_at(pos: int) -> int:
        cur = 1
        for off, n in scenes:
            if off <= pos:
                cur = n
            else:
                break
        return cur

    for i, bm in enumerate(BLOCK_RE.finditer(text)):
        kind, spk, emo = _classify(bm.group("head"))
        body = " ".join(bm.group("body").strip().split())
        ep.blocks.append(Block(
            id=f"E{ep.num:02d}-B{i + 1:03d}", ep=ep.num, scene=scene_at(bm.start()), kind=kind,
            speaker=spk, emotion=emo, start=_secs(bm.group(2), bm.group(3)),
            end=_secs(bm.group(4), bm.group(5)), text=body, line=_line_of(text, bm.start()),
        ))


CUE_RE = re.compile(r"^\s*([A-ZÀ-Þ][A-ZÀ-Þ0-9 .'\-]{1,40}?)\s*(\((?P<emo>[^)]*)\))?\s*:?\s*$")
HEADING_RE = re.compile(r"^\s*(INT\.|EXT\.|INT/EXT|EST\.|SCENE\b)", re.I)


def _parse_fallback(text: str, ep: Episode) -> None:
    """Paragraph-based parser for untimed scripts (e.g. plain or Fountain-like text)."""
    ep.timed = False
    lines = text.split("\n")
    i, n, scene = 0, 0, 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip() or re.match(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES|---)", ln.strip()):
            i += 1
            continue
        start_line = i + 1
        if HEADING_RE.match(ln):
            scene += 1
            n += 1
            ep.blocks.append(Block(f"E{ep.num:02d}-B{n:03d}", ep.num, scene, "SCENE", "", "", None, None, ln.strip(), start_line))
            i += 1
            continue
        cue = CUE_RE.match(ln)
        if cue and not ln.strip().endswith(".") and i + 1 < len(lines) and lines[i + 1].strip():
            spk = cue.group(1).strip().upper()
            emo = cue.group("emo") or ""
            j, body = i + 1, []
            while j < len(lines) and lines[j].strip():
                body.append(lines[j].strip())
                j += 1
            n += 1
            ep.blocks.append(Block(f"E{ep.num:02d}-B{n:03d}", ep.num, max(scene, 1), "DIALOGUE", spk, emo, None, None, " ".join(body), start_line))
            i = j
            continue
        j, body = i, []
        while j < len(lines) and lines[j].strip():
            body.append(lines[j].strip())
            j += 1
        n += 1
        ep.blocks.append(Block(f"E{ep.num:02d}-B{n:03d}", ep.num, max(scene, 1), "ACTION", "", "", None, None, " ".join(body), start_line))
        i = j


def parse_episode(text: str, source: str = "", fallback_num: int | None = None) -> Episode:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    num = None
    m = HEADER_RE["number"].search(text)
    if m:
        dm = re.search(r"(\d+)", m.group(1))
        if dm:
            num = int(dm.group(1))
    if num is None:
        dm = re.search(r"(?:ep(?:isode)?[-_ ]*)(\d+)", os.path.basename(source), re.I) or re.search(r"(\d+)", os.path.basename(source))
        num = int(dm.group(1)) if dm else (fallback_num or 0)
    ep = Episode(num=num, source=os.path.basename(source), raw=text)
    for key in ("logline", "continuity"):
        mm = HEADER_RE[key].search(text)
        if mm:
            setattr(ep, key, mm.group(1).strip())
    if BLOCK_RE.search(text):
        _parse_timed(text, ep)
    else:
        _parse_fallback(text, ep)
    return ep


def _read_items(path_or_bytes) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if isinstance(path_or_bytes, (bytes, bytearray)):
        zf = zipfile.ZipFile(io.BytesIO(path_or_bytes))
        for n in zf.namelist():
            base = os.path.basename(n)
            if n.lower().endswith((".txt", ".fountain")) and not base.startswith(".") and "__MACOSX" not in n:
                items.append((n, zf.read(n).decode("utf-8", errors="replace")))
    elif os.path.isdir(path_or_bytes):
        for root, _, files in os.walk(path_or_bytes):
            for f in files:
                if f.lower().endswith((".txt", ".fountain")) and not f.startswith("."):
                    p = os.path.join(root, f)
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        items.append((p, fh.read()))
    elif str(path_or_bytes).lower().endswith(".zip"):
        with open(path_or_bytes, "rb") as fh:
            return _read_items(fh.read())
    else:
        with open(path_or_bytes, encoding="utf-8", errors="replace") as fh:
            items.append((str(path_or_bytes), fh.read()))
    return items


def load_show(path_or_bytes) -> list[Episode]:
    """Load episodes from a folder, a .zip (path or bytes) or a single text file."""
    items = sorted(_read_items(path_or_bytes))
    eps = []
    for i, (src, txt) in enumerate(items):
        if not txt.strip() or os.path.basename(src).lower().startswith("readme"):
            continue
        ep = parse_episode(txt, src, fallback_num=i + 1)
        if ep.blocks:
            eps.append(ep)
    eps.sort(key=lambda e: e.num)
    return eps


def compact_episode(ep: Episode, include_camera: bool = True) -> str:
    """Token-lean text for the model: one line per block, prefixed with its id and location."""
    out = [f"### EPISODE {ep.num}"]
    if ep.logline:
        out.append(f"[E{ep.num:02d}-LOGLINE] {ep.logline}")
    if ep.continuity:
        out.append(f"[E{ep.num:02d}-NOTES] {ep.continuity}")
    for b in ep.blocks:
        if b.kind == "CAMERA" and not include_camera:
            continue
        who = f" {b.speaker}" if b.kind == "DIALOGUE" else ""
        out.append(f"[{b.id} {b.where} {b.kind}{who}] {b.text}")
    return "\n".join(out)


def block_index(eps: list[Episode]) -> dict[str, Block]:
    return {b.id: b for e in eps for b in e.all_units()}
