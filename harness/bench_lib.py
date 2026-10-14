"""Shared pieces for the bf16 saved-state benchmark (PR #2247).

Everything here talks to a running Strata server the way a client does: /v1/chat/completions for prompts,
/slots/0?action=save|restore for session files, /metrics for the conversation cache's own accounting, and the
engine's stderr log for park/resume timings. No engine code is changed and no instrumented build is used: the
timings are what a normal build reports.

    python3 bench_lib.py probe          # what the server and its logs actually report
"""
from __future__ import annotations
import json, os, re, statistics, subprocess, sys, threading, time, urllib.request

# The repository this harness sits in; STRATA_ROOT overrides it for a checkout somewhere else.
ROOT = os.environ.get("STRATA_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = os.environ.get("STRATA_URL", "http://127.0.0.1:8080")
LOG = os.path.join(ROOT, os.environ.get("STRATA_LOG", "strata-iq3_s.log"))
SESSION_DIR = os.environ.get("STRATA_SESSION_DIR", os.path.join(ROOT, "session-bench"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from needle_bench import haystack, WORDS                    # deterministic filler from this repository's own text

CHUNKS = {}                                                 # per-conversation filler, built once


# ---------------------------------------------------------------- server plumbing
def call(path, body=None, timeout=7200):
    data = json.dumps(body).encode() if body is not None else b"{}"
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path, timeout=60):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(seconds=900):
    end = time.time() + seconds
    while time.time() < end:
        try:
            if get("/health", 5).get("status", "ok") or True:
                return
        except Exception:
            time.sleep(2)
    raise SystemExit("server did not come up")


def engine_pid():
    """The --serve process, not the vision wrapper that re-execs it."""
    # any engine binary: engine-cuda12/strata, strata.pre-change, a HIP build - whatever carries --serve
    out = subprocess.run(["pgrep", "-f", "--", "--serve"], capture_output=True, text=True).stdout.split()
    for pid in out:
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace")
            if "--serve" in cmd and "server.py" not in cmd:
                return int(pid)
        except OSError:
            pass
    return None


def rss_bytes(pid):
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


class RssSampler:
    """Polls VmRSS at a fixed interval and records the max seen inside named windows.

    The window max is the operation's peak only to the resolution of the interval, and the pages the engine parks
    are pageable host memory: what the kernel has already written out is not in VmRSS. Reported as
    'incremental observed peak RSS above the steady baseline', never as a true high-water mark.
    """
    def __init__(self, pid, interval_s=0.02):
        self.pid, self.interval_s, self.stop = pid, interval_s, False
        self.samples, self.window = [], None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop:
            try:
                v = rss_bytes(self.pid)
            except (OSError, ValueError):
                v = None
            if v is not None:
                self.samples.append((time.time(), v))
                if self.window is not None:
                    self.window.append(v)
            time.sleep(self.interval_s)

    def baseline(self, n=25):
        vals = [v for _, v in self.samples[-n:]]
        return statistics.median(vals) if vals else 0

    def measure(self, fn):
        """Run fn() inside a sampling window; return (result, peak_above_baseline, samples)."""
        base = self.baseline()
        self.window = []
        try:
            out = fn()
        finally:
            peak = max(self.window) if self.window else base
            self.window = None
        return out, peak - base, len(self.samples)

    def stop_now(self):
        self.stop = True
        self.thread.join(timeout=2)


# ---------------------------------------------------------------- prompts
def filler(tokens):
    key = int(tokens)
    if key not in CHUNKS:
        CHUNKS[key] = haystack(int(tokens * 3.2))
    return CHUNKS[key]


def conversation(tag, tokens, code):
    """A long document with a code word in it, plus the question about it. `tag` keeps two conversations of the
    same length from sharing a token prefix."""
    body = filler(tokens)
    at = len(body) // 2
    doc = (f"[Session {tag}] Below is a long document, opened by this session marker.\n\n" + body[:at] +
           f"\n\n The magic code word for this session is {code}. It appears exactly once in the text above.\n\n" +
           body[at:] +
           "\n\n[End of document. Question: what is the magic code word for this session? "
           "Reply with the single word.]")
    return doc


def chat(messages, max_tokens=16):
    body = {"model": "strata", "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": messages}
    t0 = time.time()
    r = call("/v1/chat/completions", body)
    wall = time.time() - t0
    t = r.get("timings", {}) or {}
    u = r.get("usage", {}) or {}
    msg = r["choices"][0]["message"]
    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    return {"answer": text, "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
            "prompt_ms": t.get("prompt_ms"), "prompt_read": t.get("prompt_n"), "cache_n": t.get("cache_n"),
            "decode_ms": t.get("predicted_ms"), "decode_tok_s": t.get("predicted_per_second"), "wall_s": wall}


def save(name):
    t0 = time.time()
    r = call("/slots/0?action=save", {"filename": name})
    r["client_wall_s"] = time.time() - t0
    return r


def restore(name):
    t0 = time.time()
    r = call("/slots/0?action=restore", {"filename": name})
    r["client_wall_s"] = time.time() - t0
    return r


# ---------------------------------------------------------------- log reading
def log_bytes():
    try:
        return os.path.getsize(LOG)
    except OSError:
        return 0


def log_lines_since(pos):
    """New engine log lines appended since `pos`. Read-only; the engine owns the file."""
    out = []
    try:
        with open(LOG, "rb") as f:
            f.seek(pos)
            out = f.read().decode(errors="replace").splitlines()
    except OSError:
        pass
    return out


PARK = re.compile(r"parked (\d+) tokens in ([\d.]+) ms; parked=(\d+) bytes=(\d+).*snapshot_bytes=(\d+)")
RESUME = re.compile(r"conversation cache: restored (\d+) tokens \((\w+)\) in ([\d.]+) ms; parked=(\d+) bytes=(\d+)")
CKPTS = re.compile(r"(\d+) checkpoints\s*$")
PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms.*?(\d+) checkpoints\s*$")
SESSION = re.compile(r"session restored (\d+) tokens, (\d+) checkpoints, (\d+) bytes from (\S+) in ([\d.]+) ms")


def park_events(lines):
    return [dict(tokens=int(m[1]), park_ms=float(m[2]), parked=int(m[3]), cache_bytes=int(m[4]), image_bytes=int(m[5]))
            for m in (PARK.search(l) for l in lines) if m]


def resume_events(lines):
    return [dict(tokens=int(m[1]), kind=m[2], resume_ms=float(m[3]), parked=int(m[4]), cache_bytes=int(m[5]))
            for m in (RESUME.search(l) for l in lines) if m]


def prompt_events(lines):
    """One per request: how many tokens came back from the cache and how many were read, and how many
    checkpoints the engine was holding after it."""
    return [dict(tokens=int(m[1]), reused=int(m[2]), read=int(m[3]), prompt_ms=float(m[4]), checkpoints=int(m[5]))
            for m in (PROMPT.search(l) for l in lines) if m]


def session_events(lines):
    return [dict(tokens=int(m[1]), checkpoints=int(m[2]), bytes=int(m[3]), file=m[4], restore_ms=float(m[5]))
            for m in (SESSION.search(l) for l in lines) if m]


def cache_counters():
    c = get("/metrics").get("conversation_cache", {})
    return {k: c.get(k) for k in ("parked", "bytes", "evictions", "parks", "restores", "last_event", "last_tokens")}


# ---------------------------------------------------------------- statistics
def stats(values):
    v = [x for x in values if x is not None]
    if not v:
        return {}
    return {"n": len(v), "median": round(statistics.median(v), 3), "min": round(min(v), 3), "max": round(max(v), 3),
            "mean": round(statistics.fmean(v), 3)}


def pct(a, b):
    return round(100.0 * (b - a) / a, 2) if a else None


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    wait_ready()
    print(json.dumps({"engine": get("/metrics")["engine"], "cache": cache_counters(), "pid": engine_pid(),
                      "rss_mib": round(rss_bytes(engine_pid()) / 2**20, 1)}, indent=1))
    pos = log_bytes()
    chat([{"role": "user", "content": "Say OK."}], 4)
    print(json.dumps(log_lines_since(pos)[-3:], indent=1))