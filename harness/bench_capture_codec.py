"""Capture and turn latency under `--session-state-codec`, at two checkpoint spacings.

`3f04cc72` made a retained conversation checkpoint take the configured codec at the moment it is captured. The engine
has no timing line for that capture, so this measures what a client can see around it:

  * `prefill` - the first prompt of a long conversation, which is where the cache is at its largest;
  * `turn` - one short turn appended to that conversation. The engine captures a checkpoint at the start of every new
    assistant turn (docs/DETAILS.md), so every timed turn carries exactly one capture, and the difference between the
    arms at the same spacing is the cost of coding it;
  * the same at `--prompt-cache-every` 16384 (the default) and 8192, which halves the spacing and so doubles how often
    a turn also crosses a checkpoint boundary and adds a periodic capture.

Server per cell, arm order alternating between rounds, `--adapt-swaps 0 --pcie-frac 0` pinned. Engine numbers come
from the `prompt N tokens = R reused + M read in P ms, G generated in Q ms, K checkpoints` line the normal build
prints; `client wall` is the client's own clock around the request.

    ../.venv/bin/python bench_capture_codec.py                     # full sweep
    ../.venv/bin/python bench_capture_codec.py --reps 1 --warm 0 --rounds 1 --spacings 8192 --arms bf16   # smoke
"""
from __future__ import annotations
import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# bench_lib reads STRATA_URL when it is imported and bench_lazy_resume sets it when IT is imported, so the URL has to
# be in the environment before either of them loads: importing bench_lib first leaves every request pointed at 8080
# while the server is listening on the harness port, and the run waits out wait_ready in silence.
os.environ.setdefault("STRATA_URL", f"http://127.0.0.1:{os.environ.get('LAZY_PORT', '8093')}")
import bench_lib as B
import bench_lazy_resume as L

ROOT = B.ROOT
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PINNED = ["--adapt-swaps", "0", "--pcie-frac", "0"]
PREFILL_TOKENS = 31250            # a token target: ~3.2 chars/token on this model, ~31,800 tokens
TURNS = 12
# the engine's own per-request line; bench_lib's PROMPT regex stops before the generated clause
TURN_RE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms[^,]*, (\d+) generated in "
                     r"([\d.]+) ms.*?, (\d+) checkpoints")


def turn(sampler, messages, max_tokens=16):
    """One request: the client's clock, and the engine's prompt / generated / checkpoint-count figures for it."""
    (r, peak, nsamp), = [sampler.measure(lambda: B.chat(messages, max_tokens))]
    lines = B.log_lines_since(POSITION[0])
    POSITION[0] = B.log_bytes()
    rec = {"prompt_tokens": r["prompt_tokens"], "client_wall_s": round(r["wall_s"], 4),
           "answer": r["answer"][:40], "rss_increment_mib": round(peak / 2 ** 20, 1), "rss_samples": nsamp}
    if (m := TURN_RE.search("\n".join(lines))):
        rec.update(engine_prompt_tokens=int(m[1]), engine_reused=int(m[2]), engine_read=int(m[3]),
                   engine_prompt_ms=float(m[4]), engine_generated_tokens=int(m[5]),
                   engine_generated_ms=float(m[6]), checkpoints_after=int(m[7]))
    return rec


POSITION = [0]


def run_cell(arm, exe, base, spacing, warm, reps, prefill_tokens):
    L.ARMS[arm] = exe if os.path.isabs(exe) else os.path.join(ROOT, exe)
    L.BASES[arm] = base if os.path.isabs(base) else os.path.join(ROOT, base)
    L.EXTRA = PINNED + ["--prompt-cache-every", str(spacing)]
    log = f"bench-capture-{arm}-{spacing}.log"
    open(os.path.join(ROOT, log), "a").close()
    os.environ["STRATA_LOG"] = log
    B.LOG = os.path.join(ROOT, log)
    proc = L.start(arm, log)
    code = B.WORDS[(prefill_tokens // 1000) % len(B.WORDS)]
    sampler = B.RssSampler(B.engine_pid())
    POSITION[0] = B.log_bytes()
    tag = f"CP{spacing}{arm}{int(time.time())}"
    opening = turn(sampler, [{"role": "user", "content": L.doc(tag, prefill_tokens, code)}], 8)
    reply = opening["answer"]
    rec = {"arm": arm, "codec": L.codec_of(arm), "spacing": spacing, "engine": os.path.basename(L.ARMS[arm]),
           "engine_args": open(f"/proc/{B.engine_pid()}/cmdline", "rb").read().decode().split("\0"),
           "code_word": code, "prefill": opening, "turns": []}
    for i in range(warm + reps):
        q = (f"Turn {i} of a long chat: in one short sentence, without quoting the document, say what it was about "
             f"and name the magic code word for this session.")
        t = turn(sampler, [{"role": "user", "content": L.doc(tag, prefill_tokens, code)},
                           {"role": "assistant", "content": reply}, {"role": "user", "content": q}], 24)
        t["ok"] = code.lower() in (t["answer"] or "").lower()
        if i >= warm:
            rec["turns"].append(t)
    L.stop(proc)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="fp32,bf16")
    ap.add_argument("--exe", default="engine-cuda12/strata.head")
    ap.add_argument("--base-fp32", default="bench-state-codec/bench-fp32.json")
    ap.add_argument("--base-bf16", default="bench-state-codec/bench-bf16.json")
    ap.add_argument("--spacings", default="16384,8192")
    ap.add_argument("--prefill-tokens", type=int, default=PREFILL_TOKENS)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="capture_codec.json")
    a = ap.parse_args()
    bases = {"fp32": a.base_fp32, "bf16": a.base_bf16}
    arms, spacings = a.arms.split(","), [int(x) for x in a.spacings.split(",")]
    out = {"harness": "bench_capture_codec.py", "engine": a.exe,
           "engine_commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
           "pinned_args": PINNED, "warm_ups": a.warm, "measured_reps": a.reps, "turns_per_cell": a.reps,
           "note": "one retained checkpoint is captured per timed turn (docs/DETAILS.md: at the start of each new "
                   "assistant turn and every --prompt-cache-every prompt tokens); the engine has no timing line for "
                   "the capture itself, so the arms differ by what a turn costs",
           "runs": []}
    for rnd in range(a.rounds):
        order = arms if rnd % 2 == 0 else list(reversed(arms))
        for spacing in spacings:
            for arm in order:
                print(f"round {rnd} spacing {spacing} arm {arm} starting", flush=True)
                rec = run_cell(arm, a.exe, bases[arm], spacing, a.warm, a.reps, a.prefill_tokens)
                rec["round"], rec["arm_order"] = rnd, order.index(arm)
                out["runs"].append(rec)
                p, ts = rec["prefill"], rec["turns"]
                med = lambda k: sorted(t[k] for t in ts)[len(ts) // 2] if ts else None
                mx = lambda k: max((t[k] for t in ts), default=None)
                print(f"  {spacing:>5} {arm:<5} prefill prompt_ms={p.get('engine_prompt_ms')} "
                      f"tok={p.get('engine_prompt_tokens')} | turn prompt_ms med={med('engine_prompt_ms')} "
                      f"max={mx('engine_prompt_ms')} | generated_ms med={med('engine_generated_ms')} "
                      f"max={mx('engine_generated_ms')} | wall med={med('client_wall_s')} "
                      f"ok={sum(1 for t in ts if t['ok'])}/{len(ts)}", flush=True)
                B.dump(os.path.join(RESULTS, a.out), out)


if __name__ == "__main__":
    main()