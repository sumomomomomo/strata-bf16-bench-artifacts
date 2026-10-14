"""Eager whole-chain widening against lazy per-checkpoint restoration, both configured bf16.

The last externally reviewed head (`2274c1c`) widened EVERY retained checkpoint of a parked conversation when that
conversation was resumed. This branch leaves each imported snapshot in the codec it was parked in and decodes only
the one a rewind selects. `--session-state-codec` and the engine executable are both read once at start-up, so each
build is its own server session; rounds alternate which arm goes first so nothing about the page cache, the expert
tier or the drive can favour one of them.

Two workloads, both parked-resume paths of the same conversation:

  append_only   the prompt continues the parked conversation (the engine resumes it through the checkpoint at its
                own end). Nothing is rewound, so the whole cost is the mount plus one restore.
  older_ckpt    the prompt is an edit that ends at ~20,000 tokens, well behind the conversation's live state: the
                engine mounts through a retained checkpoint of the parked conversation.
  deep_rewind   the prompt ends at ~18,000 tokens and asks a different question every repetition, so no parked
                copy's near-tail checkpoint matches it and the engine mounts through the periodic checkpoint at
                16,384 tokens every time. All three restore a retained snapshot; the old build restored one that
                had already been widened at the mount, this one decodes the one it selects.

Both arms run with `--adapt-swaps 0 --pcie-frac 0`, which `docs/DETAILS.md:108` names as the way to get repeatable
runs on this box.

    python3 bench_lazy_resume.py --sizes 31250,62500,125000 --warm 2 --reps 10 --rounds 2
"""
from __future__ import annotations
import argparse, json, os, re, socket, subprocess, sys, tempfile, time

PORT = os.environ.get("LAZY_PORT", "8093")
os.environ.setdefault("STRATA_URL", f"http://127.0.0.1:{PORT}")
os.environ.setdefault("STRATA_LOG", "bench-lazy-resume.log")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as B

ROOT = B.ROOT
ARM_COMMIT = {"eager": "2274c1c", "lazy": None}      # filled in per run from git
BASE = os.path.join(ROOT, "bench-state-codec", "bench-bf16.json")   # --session-state-codec bf16; --base
                                                           # picks the codec under test
# The engines and base configs under test (--arms, --arm-exe, --arm-base), so one run can compare two builds on one
# codec or two codecs on one build. Both dicts default to the pair the lazy-restoration A/B used.
ARMS = {"eager": os.path.join(ROOT, "engine-cuda12", "strata.eager-2274c1c"),
        "lazy": os.path.join(ROOT, "engine-cuda12", "strata.lazy")}
BASES = {}
EXTRA = ["--adapt-swaps", "0", "--pcie-frac", "0"]
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# an edit that ends inside the conversation: ~20,000 tokens, above the 16,384 checkpoint and below 32,768
EDIT_TOKENS = 20000
# a deeper rewind: ~18,000 tokens with a different question every repetition, so no parked copy's near-tail
# checkpoint can match it and the engine has to mount through the checkpoint at 16,384 every time
REWIND_TOKENS = 18000
CODE_CHARS = int(EDIT_TOKENS * 3.2)
REWIND_CHARS = int(REWIND_TOKENS * 3.2)

# the two lines this branch adds; the eager build has neither
RETAINED = re.compile(r"retained (\d+) checkpoints? beside the live state, (\d+) bytes .*?\((\d+) coded, "
                      r"(\d+) bytes of it\)")
SELECTED = re.compile(r"resumed through the checkpoint at (\d+) tokens \((.*?)\) in ([\d.]+) ms; the other (\d+)")


def engine_running():
    """The engine process, found the way bench_lib finds it: a --serve process that is not the server wrapper.
    `pgrep -f engine-cuda12/strata` would also match this harness's own command line whenever an arm's engine is
    named on it with --arm-exe - which is how a two-codec run says which build each arm uses - so it reported a
    running engine after every stop and the kill below then killed the run itself."""
    return B.engine_pid() is not None


def port_open():
    """True when something is listening on the bench port: a previous arm's wrapper can linger there and the next
    arm's server exits at once if it tries to bind over it."""
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", int(PORT))) == 0


def wait_ready(proc, server_log, arm):
    """Ready, or a fast failure with the reason. The engine maps ~47 GiB of experts and locks part of them for the
    GPU, so an arm started while the last one is still releasing that dies within seconds - and with the server's
    output pointed at /dev/null the harness just slept out wait_ready's 900 s and printed nothing."""
    end = time.time() + 900
    while time.time() < end:
        if proc.poll() is not None:
            tail = "(no output)"
            if os.path.exists(server_log):
                tail = open(server_log, errors="replace").read()[-2000:]
            raise SystemExit(f"arm {arm}: server exited ({proc.returncode}) before it was ready. "
                             f"Last of {server_log}:\n{tail}")
        try:
            B.get("/health", 5)
            return
        except Exception:
            time.sleep(2)
    raise SystemExit(f"arm {arm}: server did not come up in 900 s")


def start(arm, log):
    cfg = json.load(open(BASES.get(arm, BASE)))
    cfg["exe"] = ARMS[arm]
    cfg["args"] = cfg["args"] + EXTRA
    cfg["port"] = int(PORT)
    cfg["log"] = os.path.join(ROOT, log)
    path = os.path.join(tempfile.gettempdir(), f"lazy_{arm}.json")
    open(path, "w").write(json.dumps(cfg, indent=1))
    server_log = os.path.join(ROOT, f"bench-lazy-{arm}.server.log")
    with open(server_log, "w") as out:
        proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "serve", "server.py"), "--engine", "strata",
                                 "--config", path, "--port", PORT], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    wait_ready(proc, server_log, arm)
    return proc


def stop(proc):
    proc.terminate()
    for _ in range(120):
        if not engine_running():
            break
        time.sleep(1)
    if engine_running():
        # by pid, never by pattern: `pkill -f engine-cuda12/strata` matches this process's own command line too
        pid = B.engine_pid()
        if pid:
            subprocess.run(["kill", str(pid)], capture_output=True)
    # the engine gives its locked experts back on the way out and the wrapper has to exit and free the port; the next
    # arm cannot map 47 GiB until that is finished, so wait for all three and then let the kernel settle
    for _ in range(180):
        if proc.poll() is not None and not engine_running() and not port_open():
            break
        time.sleep(1)
    time.sleep(10)


def doc(tag, tokens, code, depth=0.10):
    """The harness's own filler with a code word at `depth`; `tokens` is a token target (3.2 chars/token here)."""
    body = B.filler(tokens)
    at = int(len(body) * depth)
    return (f"[Session {tag}] Below is a long document, opened by this session marker.\n\n" + body[:at] +
            f"\n\n The magic code word for this session is {code}. It appears exactly once in the text above.\n\n" +
            body[at:] + "\n\n[End of document. Question: what is the magic code word for this session? "
                        "Reply with the single word.]")


def one(sampler, messages, want=None, max_tokens=12):
    (r, peak, nsamp), = [sampler.measure(lambda: B.chat(messages, max_tokens))]
    lines = B.log_lines_since(POSITION[0])
    POSITION[0] = B.log_bytes()
    pe, pk, rs = B.prompt_events(lines), B.park_events(lines), B.resume_events(lines)
    rec = {"prompt_tokens": r["prompt_tokens"], "engine_prompt_ms": r["prompt_ms"],
           "engine_prompt_read": r["prompt_read"], "engine_reused": r["prompt_tokens"] - (r["prompt_read"] or 0),
           "client_wall_s": round(r["wall_s"], 4),          # HTTP + queue + prompt + decode, the client's own clock
           "answer": r["answer"][:40], "checkpoints_after": pe[-1]["checkpoints"] if pe else None,
           "resume_events": rs, "park_events": pk,
           "park_ms": pk[0]["park_ms"] if pk else None,
           "resume_ms": rs[0]["resume_ms"] if rs else None,
           "resume_kind": rs[0]["kind"] if rs else None,
           "image_bytes": pk[0]["image_bytes"] if pk else None,
           "cache_bytes_after": pk[0]["cache_bytes"] if pk else None,
           "rss_increment_mib": round(peak / 2 ** 20, 1), "rss_samples": nsamp}
    for line in lines:
        if (m := RETAINED.search(line)):
            rec["retained_checkpoints"], rec["retained_bytes"] = int(m[1]), int(m[2])
            rec["retained_coded"], rec["retained_coded_bytes"] = int(m[3]), int(m[4])
        if (m := SELECTED.search(line)):
            rec["selected_checkpoint_tokens"], rec["selected_how"] = int(m[1]), m[2]
            rec["selected_ms"] = float(m[3])
            rec["checkpoints_untouched"] = int(m[4])
    if want is not None:
        rec["ok"] = want.lower() in (r["answer"] or "").lower()
    return rec


POSITION = [0]


def measure(tokens, arm, warm, reps):
    code = B.WORDS[(tokens // 1000) % len(B.WORDS)]
    a_full = doc(f"LZ{tokens}A", tokens, code)
    a_short = a_full[:CODE_CHARS] + "\n\n[Edited: the document ends here. Question: what is the magic code word " \
                                    "for this session? Reply with the single word.]"
    def a_rewind(pass_no):
        return (a_full[:REWIND_CHARS] +
                f"\n\n[Edited, pass {pass_no}: the document ends here. Question: what is the magic code word for "
                "this session? Reply with the single word.]")
    sampler = B.RssSampler(B.engine_pid())
    # set both conversations up once: the first turn reads A, the second parks it and reads the edit
    POSITION[0] = B.log_bytes()
    opening = one(sampler, [{"role": "user", "content": a_full}], code, 8)
    reply = opening["answer"]
    edit = one(sampler, [{"role": "user", "content": a_short}], code)
    rec = {"target_tokens": tokens, "arm": arm, "engine": os.path.basename(ARMS[arm]),
           "code_word": code, "prefill_tokens": opening["prompt_tokens"], "prefill_prompt_ms": opening["engine_prompt_ms"],
           "prefill_answer": opening["answer"][:40], "prefill_ok": opening.get("ok"),
           "edit_tokens": edit["prompt_tokens"], "checkpoints_after_prefill": edit["checkpoints_after"]}
    turns = []
    for i in range(warm + reps):
        append = one(sampler, [{"role": "user", "content": a_full}, {"role": "assistant", "content": reply},
                               {"role": "user", "content": "And without the document again: one short sentence "
                                                           "about it."}], code)
        rewind = one(sampler, [{"role": "user", "content": a_short}], code)
        deep = one(sampler, [{"role": "user", "content": a_rewind(i)}], code)
        for kind, t in (("append_only", append), ("older_ckpt", rewind), ("deep_rewind", deep)):
            t = dict(t); t["rep"], t["warm"], t["workload"] = i, i < warm, kind
            turns.append(t)
    sampler.stop_now()
    rec["turns"] = turns
    for kind in ("append_only", "older_ckpt", "deep_rewind"):
        cold = [t for t in turns if t["workload"] == kind and not t["warm"]]
        rec[kind] = {
            "n": len(cold),
            "answered_right": sum(1 for t in cold if t.get("ok")),
            "engine_prompt_ms": B.stats([t["engine_prompt_ms"] for t in cold]),
            "client_wall_s": B.stats([t["client_wall_s"] for t in cold]),
            "park_ms": B.stats([t["park_ms"] for t in cold if t["park_ms"] is not None]),
            "resume_ms": B.stats([t["resume_ms"] for t in cold if t["resume_ms"] is not None]),
            "image_bytes": sorted({t["image_bytes"] for t in cold if t["image_bytes"]}),
            "reused_tokens": sorted({t["engine_reused"] for t in cold}),
            "prompt_read": sorted({t["engine_prompt_read"] for t in cold}),
            "resume_kinds": sorted({t["resume_kind"] for t in cold if t["resume_kind"]}),
            "retained_checkpoints": sorted({t["retained_checkpoints"] for t in cold
                                            if "retained_checkpoints" in t}),
            "retained_bytes": sorted({t["retained_bytes"] for t in cold if "retained_bytes" in t}),
            "retained_coded": sorted({t["retained_coded"] for t in cold if "retained_coded" in t}),
            "retained_coded_bytes": sorted({t["retained_coded_bytes"] for t in cold
                                            if "retained_coded_bytes" in t}),
            "selected_checkpoint_tokens": sorted({t["selected_checkpoint_tokens"] for t in cold
                                                  if "selected_checkpoint_tokens" in t}),
            "selected_ms": B.stats([t["selected_ms"] for t in cold if "selected_ms" in t]),
            "selected_how": sorted({t["selected_how"] for t in cold if "selected_how" in t}),
            "checkpoints_after": sorted({t["checkpoints_after"] for t in cold if t["checkpoints_after"]}),
            "rss_increment_mib": B.stats([t["rss_increment_mib"] for t in cold]),
        }
    return rec


def codec_of(arm):
    """What --session-state-codec says in the arm's base config; 'default' when the option is absent."""
    args = json.load(open(BASES.get(arm, BASE)))["args"]
    return args[args.index("--session-state-codec") + 1] if "--session-state-codec" in args else "default"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="31250,62500,125000")
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--arms", default="eager,lazy")
    ap.add_argument("--base", default=os.path.join(ROOT, "bench-state-codec", "bench-bf16.json"))
    ap.add_argument("--arm-exe", action="append", default=[], metavar="ARM=PATH",
                    help="engine for one arm, so two codecs can run on one build (repeatable)")
    ap.add_argument("--arm-base", action="append", default=[], metavar="ARM=PATH",
                    help="base config for one arm, which is where --session-state-codec is set (repeatable)")
    ap.add_argument("--out", default="lazy_resume.json", help="result file under results/")
    a = ap.parse_args()
    global BASE, ARMS, BASES
    BASE = a.base
    for spec in a.arm_exe:
        name, _, path = spec.partition("=")
        ARMS[name] = path if os.path.isabs(path) else os.path.join(ROOT, path)
    for spec in a.arm_base:
        name, _, path = spec.partition("=")
        BASES[name] = path if os.path.isabs(path) else os.path.join(ROOT, path)
    sizes = [int(x) for x in a.sizes.split(",")]
    arms = a.arms.split(",")
    out = {"harness": "bench_lazy_resume.py", "config": json.load(open(BASE))["args"],
           "base_configs": {arm: BASES.get(arm, BASE) for arm in arms},
           "extra_args": EXTRA, "warm_ups": a.warm, "measured_reps": a.reps,
           "sampling_interval_s": 0.02,
           "sampling_note": "VmRSS polled every 20 ms from outside the process; the figure is the observed peak "
                            "above the steady baseline of the last 5 samples, not a high-water mark. Pages the "
                            "kernel has written out are not in VmRSS and most parked state is pageable.",
           "edit_tokens_target": EDIT_TOKENS, "engines": ARMS, "runs": []}
    for rnd in range(a.rounds):
        order = arms if rnd % 2 == 0 else list(reversed(arms))   # alternate which build goes first
        for arm in order:
            log = f"bench-lazy-{arm}.log"
            open(os.path.join(ROOT, log), "a").close()
            print(f"round {rnd} arm {arm} starting", flush=True)
            proc = start(arm, log)
            os.environ["STRATA_LOG"] = os.path.join(ROOT, log)
            B.LOG = os.path.join(ROOT, log)
            env = {"round": rnd, "order_position": order.index(arm), "arm": arm,
                   "engine": ARMS[arm], "engine_commit": ARM_COMMIT.get(arm) or
                   os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
                   "base_config": BASES.get(arm, BASE), "codec": codec_of(arm),
                   "codec_log_line": next((l.split("strata serve: ")[-1] for l in B.log_lines_since(0)
                                           if "stored conversation state" in l), None),
                   "engine_args": open(f"/proc/{B.engine_pid()}/cmdline", "rb").read().decode().split("\0"),
                   "metrics": B.get("/metrics").get("engine"), "cache": B.get("/metrics").get("conversation_cache")}
            for tokens in sizes:
                t0 = time.time()
                rec = measure(tokens, arm, a.warm, a.reps)
                rec["wall_s_for_size"] = round(time.time() - t0, 1)
                out["runs"].append(dict(env, **rec))
                B.dump(os.path.join(RESULTS, a.out), out)
                for kind in ("append_only", "older_ckpt", "deep_rewind"):
                    s = rec[kind]
                    print(f"  {tokens:>6} {arm:<6} {kind:<12} prompt_ms={s['engine_prompt_ms'].get('median')} "
                          f"wall={s['client_wall_s'].get('median')} park={s['park_ms'].get('median')} "
                          f"resume={s['resume_ms'].get('median')} kinds={s['resume_kinds']} "
                          f"reused={s['reused_tokens']} sel={s['selected_checkpoint_tokens']} "
                          f"retained={s['retained_bytes']} ok={s['answered_right']}/{s['n']}", flush=True)
            stop(proc)
    print("done", flush=True)


if __name__ == "__main__":
    main()