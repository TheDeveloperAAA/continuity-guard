"""Model client. One OpenAI-compatible interface, so swapping the model is a settings change.

Works with Gemini (default), OpenAI, Anthropic's OpenAI-compatible endpoint, Groq, OpenRouter,
local vLLM/Ollama, or any endpoint that speaks /chat/completions.
Features: JSON output, retries with backoff, fallback models, on-disk cache, token log, mock mode.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

import requests


class LLMError(RuntimeError):
    pass


REPLAY_KEY = "replay-no-key"   # api key value that means: answers come only from recorded runs


def order_free_key(model: str, system: str, user: str, temperature) -> str:
    """Cache key that ignores line order, so a reordered but otherwise identical request still hits the cache."""
    lines = sorted(line.strip() for line in user.splitlines())
    return hashlib.sha256(json.dumps([model, system, lines, temperature]).encode()).hexdigest()[:24]


def _id_signatures(prompt: str) -> dict:
    """Map each item id listed in a prompt (N01, S02, T03...) to an order-free signature of its content."""
    out, dec, i = {}, json.JSONDecoder(), 0
    while True:
        i = prompt.find("[", i)
        if i < 0:
            return out
        try:
            arr, end = dec.raw_decode(prompt, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(arr, list) and arr and all(isinstance(x, dict) and "id" in x for x in arr):
            for x in arr:
                body = {k: v for k, v in x.items() if k != "id"}
                out[x["id"]] = tuple(sorted(json.dumps(body, indent=1, sort_keys=True).splitlines()))
        i = end


def remap_ids(parsed, old_prompt: str, new_prompt: str):
    """Rewrite item ids in a saved answer so they match the same items in a reordered prompt. None if unsafe."""
    old_sig, new_sig = _id_signatures(old_prompt), _id_signatures(new_prompt)
    by_sig = {}
    for k, v in new_sig.items():
        by_sig.setdefault(v, []).append(k)
    if any(len(v) > 1 for v in by_sig.values()) or len(old_sig) != len(new_sig):
        return None
    mapping = {}
    for k, v in old_sig.items():
        if v not in by_sig:
            return None
        mapping[k] = by_sig[v][0]
    text = json.dumps(parsed)
    for k in mapping:  # two-step replace so ids can swap safely
        text = text.replace(f'"{k}"', f'"@@{k}@@"')
    for k, v in mapping.items():
        text = text.replace(f'"@@{k}@@"', f'"{v}"')
    return json.loads(text)


def parse_json(text: str):
    """Parse model output into JSON, tolerating code fences and trailing commas."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b > a:
        frag = re.sub(r",\s*([}\]])", r"\1", t[a:b + 1])
        return json.loads(frag)
    raise json.JSONDecodeError("no JSON object found", t, 0)


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    by_stage: dict = field(default_factory=dict)
    models_used: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    def add(self, stage: str, pt: int, ct: int, secs: float, cached: bool, model: str):
        s = self.by_stage.setdefault(stage, {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0})
        s["calls"] += 1
        s["cached"] += int(cached)
        s["prompt_tokens"] += pt
        s["completion_tokens"] += ct
        s["seconds"] = round(s["seconds"] + secs, 2)
        self.calls += 1
        self.cached_calls += int(cached)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.seconds = round(self.seconds + secs, 2)
        self.models_used[model] = self.models_used.get(model, 0) + 1


class LLM:
    def __init__(self, base_url: str, model: str, api_key: str, fallbacks: list[str] | None = None,
                 temperature: float = 0.0, json_mode: bool = True, timeout_s: int = 300, max_retries: int = 6,
                 min_seconds_between_calls: float = 0.0, cache_dir: str | None = None, raw_dir: str | None = None,
                 extra_body: dict | None = None, max_output_tokens: int | None = None, max_waits_429: int = 10):
        self.base_url = base_url.rstrip("/")
        self.models = [model] + [m for m in (fallbacks or []) if m and m != model]
        self.api_key = api_key
        self.temperature = temperature
        self.json_mode = json_mode
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.min_gap = min_seconds_between_calls
        self.cache_dir = cache_dir
        self.raw_dir = raw_dir
        self.extra_body = extra_body or {}
        self.max_output_tokens = max_output_tokens
        self.max_waits_429 = max_waits_429
        self.usage = Usage()
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self._n = 0
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        if raw_dir:
            os.makedirs(raw_dir, exist_ok=True)

    # ------------------------------------------------------------------ public
    def chat_json(self, system: str, user: str, stage: str = "call", payload=None) -> dict:
        """Send one request and return parsed JSON. Safe to call from several threads.
        `payload` is ignored here (used by MockLLM)."""
        with self._lock:
            self._n += 1
            n = self._n
        # saved answers are per model: a weaker model's answers are never reused for a stronger one
        key = hashlib.sha256(json.dumps([self.models[0], system, user, self.temperature]).encode()).hexdigest()[:24]
        cpath = os.path.join(self.cache_dir, f"{key}.json") if self.cache_dir else None
        # order-insensitive key: the same request with its items listed in a different order reuses the saved answer
        okey = order_free_key(self.models[0], system, user, self.temperature)
        opath = os.path.join(self.cache_dir, f"o-{okey}.json") if self.cache_dir else None
        rec = None
        if cpath and os.path.exists(cpath):
            with open(cpath, encoding="utf-8") as fh:
                rec = json.load(fh)
        elif opath and os.path.exists(opath):
            # same request, items listed in another order: reuse the answer only if every item id can be mapped
            with open(opath, encoding="utf-8") as fh:
                old = json.load(fh)
            parsed = remap_ids(old["parsed"], old["user"], user)
            if parsed is not None:
                rec = {**old, "parsed": parsed, "user": user}
        if rec is not None:
            with self._lock:
                self.usage.add(stage, 0, 0, 0.0, True, rec.get("model", "?"))
            self._save_raw(n, stage, rec, cached=True)
            print(f"  [{n:02d}] {stage}: from cache", flush=True)
            return rec["parsed"]
        last_err = None
        for model in self.models:
            try:
                text, usage, secs = self._request(model, system, user)
                try:
                    parsed = parse_json(text)
                except json.JSONDecodeError:
                    # one repair attempt: ask again, stressing JSON-only output
                    text, usage2, secs2 = self._request(model, system, user + "\n\nReturn ONLY one valid JSON object. No prose.")
                    usage = {k: usage.get(k, 0) + usage2.get(k, 0) for k in ("prompt_tokens", "completion_tokens")}
                    secs += secs2
                    parsed = parse_json(text)
                rec = {"model": model, "stage": stage, "system": system, "user": user, "response_text": text,
                       "parsed": parsed, "usage": usage, "seconds": round(secs, 2)}
                for path in (cpath, opath):
                    if path:
                        with open(path, "w", encoding="utf-8") as fh:
                            json.dump(rec, fh, ensure_ascii=False)
                with self._lock:
                    self.usage.add(stage, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), secs, False, model)
                self._save_raw(n, stage, rec, cached=False)
                print(f"  [{n:02d}] {stage}: done with {model} in {secs:.0f}s", flush=True)
                return parsed
            except (LLMError, json.JSONDecodeError) as exc:
                last_err = exc
                with self._lock:
                    self.usage.failures.append({"stage": stage, "model": model, "error": str(exc)[:300]})
                print(f"  [{n:02d}] {stage}: {model} failed ({str(exc)[:90]})" + (", trying the next model" if model != self.models[-1] else ""), flush=True)
                continue
        raise LLMError(f"All models failed for stage '{stage}': {last_err}")

    # ------------------------------------------------------------------ internals
    def _save_raw(self, n: int, stage: str, rec: dict, cached: bool):
        if not self.raw_dir:
            return
        path = os.path.join(self.raw_dir, f"{n:03d}_{stage}.json")
        out = dict(rec)
        out["from_cache"] = cached
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)

    def _request(self, model: str, system: str, user: str):
        if self.api_key == REPLAY_KEY:  # replaying a recorded run: never call the network
            raise LLMError("replay mode: no recorded answer for this request")
        body = {"model": model, "temperature": self.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.max_output_tokens:
            body["max_tokens"] = self.max_output_tokens
        body.update(self.extra_body)
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        dropped_json_mode = False
        attempt, waits_429 = 0, 0
        while True:
            with self._lock:  # space out request starts across threads
                now = time.time()
                start = max(now, self._next_slot)
                self._next_slot = start + self.min_gap
            if start > now:
                time.sleep(start - now)
            t0 = time.time()
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout_s)
            except requests.RequestException as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise LLMError(f"network error: {str(exc)[:120]}")
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            secs = time.time() - t0
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    data = data[0]
                try:
                    text = data["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError):
                    raise LLMError(f"unexpected response shape: {str(data)[:300]}")
                return text, data.get("usage") or {}, secs
            msg = r.text[:400]
            if r.status_code == 400 and not dropped_json_mode and "response_format" in body and re.search(r"response_format|json", msg, re.I):
                body.pop("response_format", None)
                dropped_json_mode = True
                continue
            if r.status_code == 429:
                # Rate limit. A daily quota means this model is done for today: move to the next model.
                # A per-minute limit means "wait": pause ALL requests for as long as the server asks, same model.
                if re.search(r"PerDay|per day|daily", r.text, re.I):
                    raise LLMError(f"HTTP 429 daily free quota used up for {model}")
                waits_429 += 1
                if waits_429 > self.max_waits_429:
                    raise LLMError(f"HTTP 429 rate limit for {model} did not clear")
                delay = _retry_delay(r) or min(60, 15 * waits_429)
                with self._lock:
                    resume = time.time() + delay
                    announce = resume > self._next_slot + 1
                    self._next_slot = max(self._next_slot, resume)
                if announce:
                    print(f"  rate limit reached on {model}: all requests pause {delay:.0f}s, then continue", flush=True)
                continue
            if r.status_code in (500, 502, 503, 504):
                attempt += 1
                if attempt <= self.max_retries:
                    time.sleep(min(60, 2 ** attempt * 3))
                    continue
            raise LLMError(f"HTTP {r.status_code} from {model}: {_short_error(msg)}")


def _retry_delay(r) -> float | None:
    ra = r.headers.get("Retry-After", "")
    if ra.replace(".", "", 1).isdigit():
        return float(ra) + 1
    m = re.search(r'retryDelay"?\s*:\s*"(\d+(?:\.\d+)?)s', r.text) or re.search(r"retry in ([\d.]+)\s*s", r.text, re.I)
    return min(120.0, float(m.group(1)) + 2) if m else None


def _short_error(text: str) -> str:
    m = re.search(r'"message"\s*:\s*"([^"]{0,160})', text)
    return m.group(1) if m else " ".join(text.split())[:160]


def resolve_api_key(env_names: list[str]) -> str:
    for n in env_names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    try:  # Google Colab secrets panel
        from google.colab import userdata  # type: ignore
        for n in env_names:
            try:
                v = (userdata.get(n) or "").strip()
                if v:
                    return v
            except Exception:
                continue
    except Exception:
        pass
    return ""


def build_llm(cfg: dict, out_dir: str | None = None, mock: bool = False, cache_dir: str | None = ".cg_cache"):
    m = cfg["model"]
    cache = cache_dir
    raw = os.path.join(out_dir, "llm_calls") if out_dir else None
    if mock:
        from .mock import MockLLM
        return MockLLM(raw_dir=raw)
    key = resolve_api_key(m.get("api_key_env", ["LLM_API_KEY", "GEMINI_API_KEY"]))
    if not key:
        raise LLMError("No API key found. Set LLM_API_KEY / GEMINI_API_KEY (or Colab secret), or use --mode code.")
    return LLM(base_url=m["base_url"], model=m["name"], api_key=key, fallbacks=m.get("fallbacks", []),
               temperature=m.get("temperature", 0), json_mode=m.get("json_mode", True), timeout_s=m.get("timeout_s", 300),
               max_retries=m.get("max_retries", 6), min_seconds_between_calls=m.get("min_seconds_between_calls", 0),
               cache_dir=cache, raw_dir=raw, extra_body=m.get("extra_body") or {}, max_output_tokens=m.get("max_output_tokens"))


def load_recorded_answers(src: str, cache_dir: str, model: str | None = None) -> int:
    """Turn the llm_calls/*.json logs of an earlier run (a folder or a raw_outputs .zip) into cache entries,
    so the pipeline can be re-run without an API key on the exact answers the model gave then."""
    import glob, tempfile, zipfile
    if src.endswith(".zip"):
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp, [n for n in z.namelist() if "/llm_calls/" in "/" + n and n.endswith(".json")])
        src = tmp
    os.makedirs(cache_dir, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(src, "**", "llm_calls", "*.json"), recursive=True)):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        rec = {k: r.get(k) for k in ("model", "stage", "system", "user", "response_text", "parsed", "usage", "seconds")}
        for m in {r["model"], model or r["model"]}:
            for path in (hashlib.sha256(json.dumps([m, r["system"], r["user"], 0]).encode()).hexdigest()[:24],
                         "o-" + order_free_key(m, r["system"], r["user"], 0)):
                with open(os.path.join(cache_dir, f"{path}.json"), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, ensure_ascii=False)
        n += 1
    return n
