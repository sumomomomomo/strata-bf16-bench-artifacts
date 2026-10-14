#!/usr/bin/env python3
"""Re-measure the RAM section on the current build: SAVE/RESTORE RSS increments at 32K and 63K, plus steady RSS and
the conversation cache's own accounting with several conversations parked.

Why this exists: the original `phase_memory` in bench_run.py writes `results/memory-{codec}.json` keyed on codec alone,
so running it once per size overwrote the first run's file. The 63K bf16 run and a 32K bf16 run were lost that way, and
four figures in docs/STATE_SNAPSHOT_BF16.md could not be reproduced from any surviving artifact. This script takes both
sizes in ONE invocation and writes a labelled file, so that cannot happen again.

Methodology is deliberately identical to phase_memory: VmRSS polled at 20 ms from outside the process, peak above the
steady baseline inside each operation window, n=3, engine already running (this attaches to it, it does not start one).
VmRSS misses pages the kernel has already written out and most parked state is pageable, so every increment here is an
observed lower bound, not a high-water mark.

Usage:  python bench_memory_rerun.py <fp32|bf16> [tokens ...]     # default 31250 62500
"""
import json
import os
import sys
import time

import bench_lib as B

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
REPS = int(os.environ.get("MEM_REPS", "3"))      # n=3 could not separate a fixed transient from noise
PARK_TAGS = ("P", "Q", "R")


def doc_frac(tag, tokens, code, frac):
    """Same helper as bench_run.doc_frac, copied because bench_run calls main() at import time."""
    body = B.filler(tokens)
    at = int(len(body) * frac)
    return (f"[Session {tag}] Below is a long document, opened by this session marker.\n\n" + body[:at] +
            f"\n\n The magic code word for this session is {code}. It appears exactly once in the text above.\n\n" +
            body[at:] + "\n\n[End of document. Question: what is the magic code word for this session? "
            "Reply with the single word.]")


def env_block(codec):
    m = B.get("/metrics")
    started = [l for l in B.log_lines_since(0) if "stored conversation state" in l]
    return {"commit": os.popen(f"git -C {B.ROOT} rev-parse HEAD").read().strip(),
            "engine_version": m["engine"].get("version"), "codec_under_test": codec,
            "codec_log_line": started[-1].split("strata serve: ")[-1] if started else None,
            "engine_args": open(f"/proc/{B.engine_pid()}/cmdline", "rb").read().decode().split("\0"),
            "model": m["engine"].get("model"), "kv": m["engine"].get("kv"),
            "max_context": m["engine"].get("max_context"), "hardware_static": m.get("hardware_static"),
            "server_url": B.URL, "log": B.LOG}


def park_three(codec, tokens, sampler, pid):
    """Three conversations at `tokens`, cycled so the cache holds more than one parked image, then steady RSS and the
    cache's own accounting. Reports the parked count it actually reached rather than assuming three."""
    rec = {"target_tokens": tokens, "prefills": []}
    docs = {}
    for i, tag in enumerate(PARK_TAGS):
        code = B.WORDS[(tokens // 1000 + i) % len(B.WORDS)]
        docs[tag] = doc_frac(f"PARK{codec}{tag}", tokens, code, 0.5)
        r = B.chat([{"role": "user", "content": docs[tag]}], 8)
        rec["prefills"].append({"tag": tag, "tokens": r["prompt_tokens"], "answer": r["answer"][:30], "want": code})
        time.sleep(1)
    # Cycle A -> B -> C -> A so each image is parked at least once and one is restored.
    cycle = []
    for tag in (PARK_TAGS[0],):
        pos = B.log_bytes()
        r = B.chat([{"role": "user", "content": docs[tag]}, {"role": "assistant", "content": "copper"},
                    {"role": "user", "content": "One short sentence about it."}], 16)
        lines = B.log_lines_since(pos)
        cycle.append({"tag": tag, "prompt_ms": r["prompt_ms"], "resume_events": B.resume_events(lines),
                      "park_events": B.park_events(lines)})
    rec["cycle"] = cycle
    time.sleep(3)                                     # let parking settle before sampling steady state
    rec["steady_rss_mib_after_parking"] = round(B.rss_bytes(pid) / 2**20, 1)
    rec["cache_after_parking"] = B.cache_counters()
    return rec


def main():
    codec = sys.argv[1]
    sizes = [int(x) for x in (sys.argv[2:] or ["31250", "62500"])]
    B.wait_ready()
    out = {"phase": "memory_rerun", "codec": codec, "env": env_block(codec), "sampling_interval_s": 0.02,
           "note": "one invocation covers every size; increments are observed peak RSS above steady baseline, "
                   "an observed lower bound because parked state is pageable", "sizes": []}
    pid = B.engine_pid()
    sampler = B.RssSampler(pid)
    for tokens in sizes:
        rec = {"target_tokens": tokens}
        docs = {}
        for tag in ("A", "B"):
            code = B.WORDS[(tokens // 1000 + "AB".index(tag)) % len(B.WORDS)]
            docs[tag] = doc_frac(f"MEM{tokens}{tag}", tokens, code, 0.5)
            r = B.chat([{"role": "user", "content": docs[tag]}], 8)
            rec[f"prefill_{tag}"] = {"tokens": r["prompt_tokens"], "answer": r["answer"][:30], "want": code}
            time.sleep(1)
        fname = f"memrerun-{codec}-{tokens}.bin"
        # The first save after a prefill is not like the later ones: the allocator admits a fixed scratch buffer once
        # and every later save reuses it, so a median over reps mixes "caught the allocation" with "reused it". Measure
        # the cold case on its own, then warm up and measure steady-state reuse over REPS reps.
        cold_base = sampler.baseline()
        (rc, cold_peak, _), = [sampler.measure(lambda: B.save(fname))]
        rec["cold_save_increment_mib"] = round(cold_peak / 2**20, 1)
        rec["cold_save_baseline_mib"] = round(cold_base / 2**20, 1)
        rec["cold_save_engine_ms"] = rc["timings"]["save_ms"]
        rec["cold_save_bytes"] = rc["n_written"]
        windows = []
        for i in range(REPS):
            for op in ("save", "restore"):
                base = sampler.baseline()
                cache_before = B.cache_counters()
                if op == "save":
                    (r, peak, n), = [sampler.measure(lambda: B.save(fname))]
                    detail = {"engine_ms": r["timings"]["save_ms"], "bytes_written": r["n_written"],
                              "client_wall_s": round(r["client_wall_s"], 4)}
                else:
                    (r, peak, n), = [sampler.measure(lambda: B.restore(fname))]
                    detail = {"engine_ms": r["timings"]["restore_ms"], "bytes_read": r["n_read"],
                              "client_wall_s": round(r["client_wall_s"], 4)}
                windows.append({"rep": i, "op": op, "rss_baseline_mib": round(base / 2**20, 1),
                                "rss_peak_increment_mib": round(peak / 2**20, 1),
                                "rss_after_mib": round(B.rss_bytes(pid) / 2**20, 1),
                                "cache_before": cache_before, "cache_after": B.cache_counters(), **detail})
        rec["windows"] = windows
        rec["steady_rss_mib"] = round(sampler.baseline() / 2**20, 1)
        for op in ("save", "restore"):
            w = [x for x in windows if x["op"] == op]
            rec[f"{op}_increment_mib"] = B.stats([x["rss_peak_increment_mib"] for x in w])
            rec[f"{op}_engine_ms"] = B.stats([x["engine_ms"] for x in w])
        rec["warm_save_reps"] = [x["rss_peak_increment_mib"] for x in windows if x["op"] == "save"]
        rec["restore_reps"] = [x["rss_peak_increment_mib"] for x in windows if x["op"] == "restore"]
        # Every parked image count seen during this size's windows, so the doc can quote what was actually reached.
        rec["parked_counts_seen"] = sorted({x["cache_after"].get("parked") for x in windows if x["cache_after"]})
        rec["max_parked_bytes"] = max((x["cache_after"].get("bytes") or 0) for x in windows)
        if tokens == sizes[0]:                          # park-cycle accounting once, at the shorter size
            rec["park_cycle"] = park_three(codec, tokens, sampler, pid)
        out["sizes"].append(rec)
        print(json.dumps({k: rec.get(k) for k in ("target_tokens", "steady_rss_mib", "cold_save_increment_mib",
                                                  "save_increment_mib", "restore_increment_mib")}), flush=True)
    sampler.stop_now()
    path = os.path.join(RESULTS, f"memory-{codec}.head.json")
    B.dump(path, out)
    print("wrote", path, flush=True)


main()