"""One phase of the bf16 saved-state benchmark against a running server.

    python3 bench_run.py save_restore fp32  --sizes 4000,32000,64000,128000
    python3 bench_run.py park_resume  bf16  --sizes 32000,64000,128000
    python3 bench_run.py functional   fp32

Timings come from a normal build: the engine's own save_ms / restore_ms, its park / resume log lines, and the
client's own wall clock, kept apart. RSS windows are sampled at 20 ms from /proc and are labelled as observed
increments above a steady baseline, not as high-water marks.
"""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_lib as B

ROOT = B.ROOT
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def env_block(codec):
    m = B.get("/metrics")
    lines = B.log_lines_since(0)
    started = [l for l in lines if "stored conversation state" in l]
    return {"commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
            "engine_version": m["engine"].get("version"), "codec_under_test": codec,
            "codec_log_line": started[-1].split("strata serve: ")[-1] if started else None,
            "engine_args": open(f"/proc/{B.engine_pid()}/cmdline", "rb").read().decode().split("\0"),
            "model": m["engine"].get("model"), "kv": m["engine"].get("kv"),
            "max_context": m["engine"].get("max_context"), "cache": m["engine"],
            "hardware_static": m.get("hardware_static"), "server_url": B.URL, "log": B.LOG}


# ------------------------------------------------------------------ phase: save / restore
def phase_save_restore(codec, sizes, warm, reps):
    out = {"phase": "save_restore", "codec": codec, "env": env_block(codec), "sizes": []}
    sampler = B.RssSampler(B.engine_pid())
    for tokens in sizes:
        tag, code = f"SR{tokens}", B.WORDS[(tokens // 1000) % len(B.WORDS)]
        rec = {"target_tokens": tokens}
        r = B.chat([{"role": "user", "content": B.conversation(tag, tokens, code)}], 8)
        rec["actual_tokens"], rec["first_answer"] = r["prompt_tokens"], r["answer"][:40]
        rec["prefill_prompt_ms"], rec["prefill_wall_s"] = r["prompt_ms"], round(r["wall_s"], 2)
        rec["checkpoints_after_prefill"] = (B.prompt_events(B.log_lines_since(0)) or [{}])[-1].get("checkpoints")
        fname = f"sr-{codec}-{tokens}.bin"
        path = os.path.join(B.SESSION_DIR, fname)

        # All SAVEs first: a RESTORE puts the file's state back into the device session, so a save taken after a
        # restore of a coded file would copy already-rounded state. With the saves first, every repetition of both
        # arms reads the same fp32 live state off the device.
        saves = []
        for i in range(warm + reps):
            pos = B.log_bytes()
            s = B.save(fname)
            ev = B.session_events(B.log_lines_since(pos))     # nothing on a save; kept in case it grows
            saves.append({"rep": i, "warm": i < warm, "engine_save_ms": s["timings"]["save_ms"],
                          "client_wall_s": round(s["client_wall_s"], 4), "n_saved": s["n_saved"],
                          "n_written": s["n_written"], "bytes_on_disk": os.path.getsize(path),
                          "file_version": None})
        rec["save_file"] = fname
        rec["save_file_bytes"] = os.path.getsize(path)
        rec["save_file_version"] = int.from_bytes(open(path, "rb").read(12)[8:12], "little")
        rec["saves"] = saves
        cold = [s for s in saves if not s["warm"]]
        rec["save_engine_ms"] = B.stats([s["engine_save_ms"] for s in cold])
        rec["save_client_s"] = B.stats([s["client_wall_s"] for s in cold])

        restores = []
        for i in range(warm + reps):
            rs = B.restore(fname)
            restores.append({"rep": i, "warm": i < warm, "engine_restore_ms": rs["timings"]["restore_ms"],
                             "client_wall_s": round(rs["client_wall_s"], 4), "n_restored": rs["n_restored"],
                             "n_read": rs["n_read"]})
        rec["restores"] = restores
        cold = [s for s in restores if not s["warm"]]
        rec["restore_engine_ms"] = B.stats([s["engine_restore_ms"] for s in cold])
        rec["restore_client_s"] = B.stats([s["client_wall_s"] for s in cold])

        # every restore above ran with the file in the page cache: it was written seconds ago and read back twelve
        # times. Cold-cache numbers are a separate experiment (bench_cold.py), not mixed into these.
        rec["cache_state"] = "warm page cache (the file was written and re-read within the same run)"
        out["sizes"].append(rec)
        print(json.dumps({k: rec[k] for k in ("target_tokens", "actual_tokens", "save_file_bytes",
                                              "save_engine_ms", "restore_engine_ms")}), flush=True)
    sampler.stop_now()
    B.dump(os.path.join(RESULTS, f"save_restore-{codec}.json"), out)


# ------------------------------------------------------------------ phase: park / resume
def phase_park_resume(codec, sizes, warm, reps):
    out = {"phase": "park_resume", "codec": codec, "env": env_block(codec), "sizes": []}
    sampler = B.RssSampler(B.engine_pid())
    for tokens in sizes:
        rec = {"target_tokens": tokens}
        docs, codes = {}, {}
        for tag in ("A", "B"):
            codes[tag] = B.WORDS[("AB".index(tag) + tokens // 1000) % len(B.WORDS)]
            docs[tag] = B.conversation(f"PR{tokens}{tag}", tokens, codes[tag])
            r = B.chat([{"role": "user", "content": docs[tag]}], 8)
            rec[f"prefill_{tag}"] = {"tokens": r["prompt_tokens"], "prompt_ms": r["prompt_ms"],
                                     "wall_s": round(r["wall_s"], 2), "answer": r["answer"][:40]}
            time.sleep(1)
        ev = B.prompt_events(B.log_lines_since(0))
        rec["checkpoints_after_prefill"] = ev[-1]["checkpoints"] if ev else None
        rec["cache_after_prefill"] = B.cache_counters()

        # switch A -> B -> A ... : each request re-sends its own conversation, so its first tokens are the parked
        # conversation's own and the engine resumes it instead of reading anything again. Parking the other one is
        # what the same request does on the way in.
        turns, cur = [], "B"
        for i in range(warm + reps):
            for cur in ("A", "B"):
                ans = rec[f"prefill_{cur}"]["answer"][:40]
                pos = B.log_bytes()
                before = B.cache_counters()
                (r, peak, nsamp), = [sampler.measure(lambda: B.chat(
                    [{"role": "user", "content": docs[cur]}, {"role": "assistant", "content": ans},
                     {"role": "user", "content": "And without looking at the document again: what was the magic "
                                                 "code word for this session? Reply with the single word."}], 12))]
                lines = B.log_lines_since(pos)
                turns.append({"rep": i, "warm": i < warm, "switch_to": cur, "expected": codes[cur],
                              "answer": r["answer"], "ok": codes[cur].lower() in r["answer"].lower(),
                              "prompt_tokens": r["prompt_tokens"], "cache_n": r["cache_n"],
                              "prompt_read": r["prompt_read"], "prompt_ms": r["prompt_ms"],
                              "wall_s": round(r["wall_s"], 4),
                              "resume_events": B.resume_events(lines), "park_events": B.park_events(lines),
                              "prompt_events": B.prompt_events(lines),
                              "cache_before": before, "cache_after": B.cache_counters(),
                              "rss_peak_increment_mib": round(peak / 2**20, 1)})
        cold = [t for t in turns if not t["warm"]]
        rec["switches_total"], rec["switches_ok"] = len(cold), sum(1 for t in cold if t["ok"])
        rec["resume_ms"] = B.stats([t["resume_events"][0]["resume_ms"] for t in cold if t["resume_events"]])
        rec["park_ms"] = B.stats([t["park_events"][0]["park_ms"] for t in cold if t["park_events"]])
        rec["ttft_prompt_ms"] = B.stats([t["prompt_ms"] for t in cold])
        rec["switch_wall_s"] = B.stats([t["wall_s"] for t in cold])
        rec["resume_kinds"] = sorted({t["resume_events"][0]["kind"] for t in cold if t["resume_events"]})
        rec["image_bytes_parked"] = sorted({t["park_events"][0]["image_bytes"] for t in cold if t["park_events"]})
        rec["cache_bytes_final"] = cold[-1]["cache_after"]
        rec["rss_peak_increment_mib"] = B.stats([t["rss_peak_increment_mib"] for t in cold])
        rec["turns"] = turns
        out["sizes"].append(rec)
        print(json.dumps({k: rec[k] for k in ("target_tokens", "switches_ok", "switches_total", "resume_ms",
                                              "park_ms", "ttft_prompt_ms", "image_bytes_parked")}), flush=True)
    sampler.stop_now()
    B.dump(os.path.join(RESULTS, f"park_resume-{codec}.json"), out)


# ------------------------------------------------------------------ phase: functional
def phase_functional(codec, depths, seeds, needle_tokens, deep_tokens):
    out = {"phase": "functional", "codec": codec, "env": env_block(codec), "checks": {}}
    ok = lambda code, ans: code.lower() in (ans or "").lower()

    # 1. needle recall across a disk restore
    disk = []
    for depth in depths:
        for seed in range(seeds):
            code = B.WORDS[(depth + seed * 5) % len(B.WORDS)]
            body = B.filler(needle_tokens)
            at = int(len(body) * depth / 100)
            doc = (f"[Seed {seed}] " + body[:at] + f"\n\n The magic code word for this session is {code}.\n\n" +
                   body[at:] + "\n\nQuestion: what is the magic code word? Reply with the single word.")
            first = B.chat([{"role": "user", "content": doc}], 12)
            s = B.save(f"fn-{codec}-{needle_tokens}-{depth}-{seed}.bin")
            r = B.restore(f"fn-{codec}-{needle_tokens}-{depth}-{seed}.bin")
            again = B.chat([{"role": "user", "content": doc}, {"role": "assistant", "content": first["answer"]},
                            {"role": "user", "content": "Without the document: what was the magic code word? "
                                                        "One word."}], 12)
            disk.append({"tokens": first["prompt_tokens"], "depth": depth, "seed": seed, "want": code,
                         "cold_answer": first["answer"], "restored_answer": again["answer"],
                         "cold_ok": ok(code, first["answer"]), "restored_ok": ok(code, again["answer"]),
                         "saved_bytes": s["n_written"], "restored_tokens": r["n_restored"],
                         "restore_ms": r["timings"]["restore_ms"]})
            print("disk   ", disk[-1]["tokens"], f"d{depth}", "seed", seed, "want", code,
                  "restored", again["answer"][:24], "OK" if disk[-1]["restored_ok"] else "MISS", flush=True)
    out["checks"]["needle_across_disk_restore"] = disk

    # 2. needle recall across a parked-conversation resume
    parked = []
    for depth in depths:
        for seed in range(seeds):
            code = B.WORDS[(depth + seed * 5 + 2) % len(B.WORDS)]
            other = B.WORDS[(depth + seed * 5 + 7) % len(B.WORDS)]
            doc = B.conversation(f"FN{depth}{seed}", needle_tokens, code)
            other_doc = B.conversation(f"FN{depth}{seed}OTHER", needle_tokens // 2, other)
            first = B.chat([{"role": "user", "content": doc}], 12)
            B.chat([{"role": "user", "content": other_doc}], 8)              # forces the first one to be parked
            time.sleep(1)
            pos = B.log_bytes()
            again = B.chat([{"role": "user", "content": doc}, {"role": "assistant", "content": first["answer"]},
                            {"role": "user", "content": "Without the document: what was the magic code word? "
                                                        "One word."}], 12)
            ev = B.resume_events(B.log_lines_since(pos))
            parked.append({"tokens": first["prompt_tokens"], "depth": depth, "seed": seed, "want": code,
                           "resumed_answer": again["answer"], "ok": ok(code, again["answer"]),
                           "resume_events": ev, "cache_n": again["cache_n"], "prompt_read": again["prompt_read"]})
            print("parked ", parked[-1]["tokens"], f"d{depth}", "seed", seed, "want", code,
                  "resumed", again["answer"][:24], "OK" if parked[-1]["ok"] else "MISS",
                  "resumed_from", [e["kind"] for e in ev], flush=True)
    out["checks"]["needle_across_parked_resume"] = parked

        # 3. a real resume from an OLDER retained checkpoint: edit the prompt back to a shorter prefix of the same
    #    document, so the engine can only continue from a checkpoint that is not the newest one. The log's
    #    "restored N tokens (checkpoint)" line names which one it took.
    old = []
    for cut in (0.5, 0.25):
        code = B.WORDS[6]
        doc = B.conversation(f"OLD{int(cut*100)}", deep_tokens, code)
        first = B.chat([{"role": "user", "content": doc}], 8)
        short = (doc[:int(len(doc) * cut)] +
                 "\n\n[Edited: the document ends here. Question: what is the magic code word for this session? "
                 "Reply with the single word.]")
        pos = B.log_bytes()
        again = B.chat([{"role": "user", "content": short}], 12)
        lines = B.log_lines_since(pos)
        old.append({"full_tokens": first["prompt_tokens"], "cut": cut, "edited_tokens": again["prompt_tokens"],
                    "want": code, "answer": again["answer"], "ok": ok(code, again["answer"]),
                    "resume_events": B.resume_events(lines), "prompt_events": B.prompt_events(lines)})
        print("older-ck", "full", first["prompt_tokens"], "edited", again["prompt_tokens"],
              "resumed", [(e["tokens"], e["kind"]) for e in old[-1]["resume_events"]],
              "answer", again["answer"][:24], flush=True)
    out["checks"]["older_checkpoint_resume"] = old

    # 4. ten park -> resume -> generate cycles, with new tokens generated between them. Two conversations, each with
    #    its own history, alternated: every switch parks one and resumes the other, and each turn makes the model
    #    generate new tokens first.
    cycles = []
    code, other_code = B.WORDS[9], B.WORDS[10]
    docs = {"A": B.conversation("CYCA", needle_tokens, code), "B": B.conversation("CYCB", needle_tokens // 2, other_code)}
    qs = {"A": "What is the magic code word for this session? One word.",
          "B": "What is the magic code word for this session? One word."}
    hist = {"A": [], "B": []}
    for i in range(12):
        cur = "A" if i % 2 == 0 else "B"
        want = code if cur == "A" else other_code
        turn = ("What is the magic code word for this session? One word." if i == 0
                else "And one short sentence about it.")
        pos = B.log_bytes()
        r = B.chat(hist[cur] + [{"role": "user", "content": docs[cur] if i == 0 else turn}], 24)
        lines = B.log_lines_since(pos)
        hist[cur] += [{"role": "user", "content": docs[cur] if i == 0 else turn},
                      {"role": "assistant", "content": r["answer"]}]
        cycles.append({"cycle": i, "conversation": cur, "want": want, "answer": r["answer"],
                       "ok": want.lower() in (r["answer"] or "").lower(), "generated_tokens": r["completion_tokens"],
                       "prompt_tokens": r["prompt_tokens"], "prompt_read": r["prompt_read"], "cache_n": r["cache_n"],
                       "resume_events": B.resume_events(lines), "park_events": B.park_events(lines),
                       "cache": B.cache_counters()})
        print("cycle   ", i, cur, "resumed", [(e["tokens"], e["kind"]) for e in cycles[-1]["resume_events"]],
              "read", r["prompt_read"], "answer", r["answer"][:24], "OK" if cycles[-1]["ok"] else "MISS", flush=True)
    out["checks"]["park_resume_generate_cycles"] = cycles

    B.dump(os.path.join(RESULTS, f"functional-{codec}.json"), out)


# ------------------------------------------------------------------ phase: resume paths
def doc_frac(tag, tokens, code, frac):
    """The same long document with its code word at `frac` of its length, so a prompt edited back to a shorter
    prefix still contains it."""
    body = B.filler(tokens)
    at = int(len(body) * frac)
    return (f"[Session {tag}] Below is a long document, opened by this session marker.\n\n" + body[:at] +
            f"\n\n The magic code word for this session is {code}. It appears exactly once in the text above.\n\n" +
            body[at:] + "\n\n[End of document. Question: what is the magic code word for this session? "
            "Reply with the single word.]")


def phase_resume_paths(codec, deep_tokens, needle_tokens):
    out = {"phase": "resume_paths", "codec": codec, "env": env_block(codec), "checks": {}}
    ok = lambda code, ans: code.lower() in (ans or "").lower()

    # a real resume from an OLDER retained checkpoint. The code word sits at 10% of the document, so every edit
    # below still contains it; the edited prompt is shorter than the newest checkpoint, so the engine can only
    # continue from an older one. `reused` names which: the periodic checkpoints sit at multiples of
    # --prompt-cache-every (16384 here), so reused == 16384 means the checkpoint at that boundary was taken.
    old = []
    for cut in (0.35, 0.6):
        code = B.WORDS[6]
        doc = doc_frac(f"OLD{int(cut * 100)}", deep_tokens, code, 0.10)
        first = B.chat([{"role": "user", "content": doc}], 8)
        short = (doc[:int(len(doc) * cut)] +
                 "\n\n[Edited: the document ends here. Question: what is the magic code word for this session? "
                 "Reply with the single word.]")
        pos = B.log_bytes()
        again = B.chat([{"role": "user", "content": short}], 12)
        lines = B.log_lines_since(pos)
        pe = B.prompt_events(lines)
        old.append({"full_tokens": first["prompt_tokens"], "cut": cut, "edited_tokens": again["prompt_tokens"],
                    "want": code, "cold_answer": first["answer"], "answer": again["answer"],
                    "ok": ok(code, again["answer"]), "reused": pe[-1]["reused"] if pe else None,
                    "read": pe[-1]["read"] if pe else None, "checkpoints_after": pe[-1]["checkpoints"] if pe else None,
                    "resume_events": B.resume_events(lines), "prompt_ms": again["prompt_ms"]})
        print("older-ck", "full", first["prompt_tokens"], "edited", again["prompt_tokens"],
              "reused", old[-1]["reused"], "read", old[-1]["read"], "answer", again["answer"][:24],
              "OK" if old[-1]["ok"] else "MISS", flush=True)
    out["checks"]["older_checkpoint_resume"] = old

    # ten park -> resume -> generate cycles. Two conversations, each with its own history, alternated: every switch
    # parks one and resumes the other, and each turn makes the model generate new tokens first.
    cycles = []
    code, other_code = B.WORDS[9], B.WORDS[10]
    docs = {"A": doc_frac("CYCA", needle_tokens, code, 0.5), "B": doc_frac("CYCB", needle_tokens // 2, other_code, 0.5)}
    hist = {"A": [], "B": []}
    for i in range(12):
        cur = "A" if i % 2 == 0 else "B"
        want = code if cur == "A" else other_code
        opening = i in (0, 1)
        turn = (docs[cur] + "\n\nWhat is the magic code word for this session? One word.") if opening \
            else "And one short sentence about it."
        pos = B.log_bytes()
        r = B.chat(hist[cur] + [{"role": "user", "content": turn}], 24)
        lines = B.log_lines_since(pos)
        hist[cur] += [{"role": "user", "content": turn}, {"role": "assistant", "content": r["answer"]}]
        cycles.append({"cycle": i, "conversation": cur, "opening": opening, "want": want, "answer": r["answer"],
                       "ok": want.lower() in (r["answer"] or "").lower(), "generated_tokens": r["completion_tokens"],
                       "prompt_tokens": r["prompt_tokens"], "prompt_read": r["prompt_read"], "cache_n": r["cache_n"],
                       "resume_events": B.resume_events(lines), "park_events": B.park_events(lines),
                       "cache": B.cache_counters()})
        print("cycle   ", i, cur, "resumed", [(e["tokens"], e["kind"]) for e in cycles[-1]["resume_events"]],
              "read", r["prompt_read"], "answer", r["answer"][:24], "OK" if cycles[-1]["ok"] else "MISS", flush=True)
    out["checks"]["park_resume_generate_cycles"] = cycles
    B.dump(os.path.join(RESULTS, f"resume_paths-{codec}.json"), out)


# ------------------------------------------------------------------ phase: memory
def phase_memory(codec, sizes, reps):
    """RSS windows around the four operations, plus the conversation cache's own accounting.

    The sampler polls VmRSS at 20 ms. VmRSS does not include pages the kernel has already written out, and much of
    what Strata parks is pageable host memory, so 'increment' is an observed lower bound on the operation's
    footprint, not a high-water mark. The process's lifetime peak is deliberately not reported.
    """
    out = {"phase": "memory", "codec": codec, "env": env_block(codec), "sampling_interval_s": 0.02, "sizes": []}
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
        fname = f"mem-{codec}-{tokens}.bin"
        path = os.path.join(B.SESSION_DIR, fname)
        windows = []
        for i in range(reps):
            for op in ("save", "restore", "switch_to_B", "switch_to_A"):
                base = sampler.baseline()
                cache_before = B.cache_counters()
                if op == "save":
                    (r, peak, n), = [sampler.measure(lambda: B.save(fname))]
                    detail = {"engine_ms": r["timings"]["save_ms"], "bytes_written": r["n_written"],
                              "client_wall_s": round(r["client_wall_s"], 4)}
                elif op == "restore":
                    (r, peak, n), = [sampler.measure(lambda: B.restore(fname))]
                    detail = {"engine_ms": r["timings"]["restore_ms"], "bytes_read": r["n_read"],
                              "client_wall_s": round(r["client_wall_s"], 4)}
                else:
                    tag = op[-1]
                    ans = rec[f"prefill_{tag}"]["answer"][:30]
                    pos = B.log_bytes()
                    (r, peak, n), = [sampler.measure(lambda: B.chat(
                        [{"role": "user", "content": docs[tag]}, {"role": "assistant", "content": ans},
                         {"role": "user", "content": "One short sentence about it."}], 16))]
                    lines = B.log_lines_since(pos)
                    detail = {"prompt_ms": r["prompt_ms"], "wall_s": round(r["wall_s"], 4),
                              "resume_events": B.resume_events(lines), "park_events": B.park_events(lines)}
                windows.append({"rep": i, "op": op, "rss_baseline_mib": round(base / 2**20, 1),
                                "rss_peak_increment_mib": round(peak / 2**20, 1),
                                "rss_after_mib": round(B.rss_bytes(pid) / 2**20, 1),
                                "cache_before": cache_before, "cache_after": B.cache_counters(), **detail})
        rec["windows"] = windows
        rec["steady_rss_mib"] = round(sampler.baseline() / 2**20, 1)
        for op in ("save", "restore", "switch_to_B", "switch_to_A"):
            w = [x for x in windows if x["op"] == op]
            rec[f"{op}_increment_mib"] = B.stats([x["rss_peak_increment_mib"] for x in w])
            if "engine_ms" in w[0]:
                rec[f"{op}_engine_ms"] = B.stats([x["engine_ms"] for x in w])
            else:
                rec[f"{op}_engine_ms"] = B.stats([(x["resume_events"] or [{}])[0].get("resume_ms") for x in w])
                rec[f"{op}_park_ms"] = B.stats([(x["park_events"] or [{}])[0].get("park_ms") for x in w])
        rec["cache_bytes_after"] = B.cache_counters()
        out["sizes"].append(rec)
        print(json.dumps({k: rec[k] for k in ("target_tokens", "steady_rss_mib", "save_increment_mib",
                                              "restore_increment_mib", "switch_to_A_increment_mib")}), flush=True)
    sampler.stop_now()
    B.dump(os.path.join(RESULTS, f"memory-{codec}.json"), out)


# ------------------------------------------------------------------ phase: needle at one length, disk restore only
def phase_needle(codec, tokens, depths, seeds):
    out = {"phase": "needle", "codec": codec, "env": env_block(codec), "target_tokens": tokens, "runs": []}
    for depth in depths:
        for seed in range(seeds):
            code = B.WORDS[(depth + seed * 5) % len(B.WORDS)]
            body = B.filler(tokens)
            at = int(len(body) * depth / 100)
            doc = (f"[Seed {seed}] " + body[:at] + f"\n\n The magic code word for this session is {code}.\n\n" +
                   body[at:] + "\n\nQuestion: what is the magic code word? Reply with the single word.")
            first = B.chat([{"role": "user", "content": doc}], 12)
            s = B.save(f"nd-{codec}-{tokens}-{depth}-{seed}.bin")
            r = B.restore(f"nd-{codec}-{tokens}-{depth}-{seed}.bin")
            again = B.chat([{"role": "user", "content": doc}, {"role": "assistant", "content": first["answer"]},
                            {"role": "user", "content": "Without the document: what was the magic code word? "
                                                        "One word."}], 12)
            out["runs"].append({"tokens": first["prompt_tokens"], "depth": depth, "seed": seed, "want": code,
                                "cold_answer": first["answer"], "restored_answer": again["answer"],
                                "cold_ok": code.lower() in (first["answer"] or "").lower(),
                                "restored_ok": code.lower() in (again["answer"] or "").lower(),
                                "saved_bytes": s["n_written"], "restore_ms": r["timings"]["restore_ms"],
                                "resumed_prompt_read": again["prompt_read"]})
            print("needle ", out["runs"][-1]["tokens"], f"d{depth}s{seed}", "want", code,
                  "after restore", again["answer"][:24], "OK" if out["runs"][-1]["restored_ok"] else "MISS", flush=True)
    ok = sum(1 for x in out["runs"] if x["restored_ok"])
    out["summary"] = {"attempts": len(out["runs"]), "success": ok}
    B.dump(os.path.join(RESULTS, f"needle-{tokens}-{codec}.json"), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["save_restore", "park_resume", "functional", "resume_paths", "memory", "needle"])
    ap.add_argument("codec", choices=["fp32", "bf16"])
    ap.add_argument("--sizes", default="4000,32000,64000,128000")
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--depths", default="10,50,90")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--needle-tokens", type=int, default=32000)
    ap.add_argument("--deep-tokens", type=int, default=64000)
    a = ap.parse_args()
    B.wait_ready()
    os.environ.setdefault("STRATA_LOG", "strata-iq3_s.log")
    if a.phase == "save_restore":
        phase_save_restore(a.codec, [int(x) for x in a.sizes.split(",")], a.warm, a.reps)
    elif a.phase == "park_resume":
        phase_park_resume(a.codec, [int(x) for x in a.sizes.split(",")], a.warm, a.reps)
    elif a.phase == "needle":
        phase_needle(a.codec, a.needle_tokens, [int(x) for x in a.depths.split(",")], a.seeds)
    elif a.phase == "memory":
        phase_memory(a.codec, [int(x) for x in a.sizes.split(",")], a.reps)
    elif a.phase == "resume_paths":
        phase_resume_paths(a.codec, a.deep_tokens, a.needle_tokens)
    else:
        phase_functional(a.codec, [int(x) for x in a.depths.split(",")], a.seeds, a.needle_tokens, a.deep_tokens)


main()