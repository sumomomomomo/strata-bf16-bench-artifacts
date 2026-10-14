# bf16 saved-state benchmark

Measures PR #2247 (`--session-state-codec fp32|bf16`): how much less RAM and disk a parked conversation and a
session file take, what it costs in milliseconds, and whether conversations still answer correctly after a restore.

Nothing here is installed by the PR and nothing here edits the PR. The engine under test is a build of the PR head
in `build-cuda12/`, deployed to `engine-cuda12/`; each comparison names the two builds or the two configs it ran.
All timings come from those normal builds; no instrumented binary was used. Memory is sampled from outside the
process (`/proc/<pid>/status`, 20 ms).

## Machine

Linux Mint 22.1, kernel 6.14.0-33-generic, Genuine Intel(R) 0000 (24 cores / 32 threads), 93 GB RAM,
Tesla V100-PCIE-32GB (CUDA arch 70, driver 580.178.04, toolkit `/usr/local/cuda-12.8`), an RTX 2080 Ti that the
engine does not use, Samsung 990 EVO Plus 4TB on ext4. Model `qwen3.8-flash-next-iq3_s`, K/V `int8`,
`--max-context 262144`, `--conversation-cache-mib 16384 --conversation-cache-slots 12
--conversation-cache-min-free-mib 8192`, `--prompt-cache 6` and `--prompt-cache-every 16384` (both defaults).

## Why two server processes instead of one

`--session-state-codec` is read once at start-up and decides how every parked image and every session file is
written; there is no per-request switch. So each arm is its own engine process, started from the same binary with
configs that differ in exactly three keys (`--session-state-codec`, the engine log path, and `slot_save_path`, which
only turns the SAVE/RESTORE HTTP commands on and is set for both arms). Same prompts, same files, same order, and
each phase runs the arms in the same sequence. Alternating one conversation between codecs inside a single process
is not possible with this PR.

## Files

| file | what it is |
| --- | --- |
| `bench_lib.py` | server/client helpers, the 20 ms RSS sampler, log parsers for the engine's PARK / RESUME / PROMPT / SESSION lines |
| `bench_run.py` | the five phases: `save_restore`, `park_resume`, `functional`, `resume_paths`, `memory`, plus `needle` |
| `bench_cold.py` | cold-page-cache restore: `posix_fadvise(DONTNEED)` on one file, then one timed restore |
| `bench_encode.cpp` | host-only encode/decode timing on one real 118 MB running state |
| `extract_state.py` | pulls that state's bytes out of a session file this build wrote |
| `bench_report.py` | turns `results/*.json` into `results/REPORT.md` |
| `bench_convert_host.cpp` | the stored-to-stored conversion timed on a real state, allocation included: the path in the header against a verbatim copy of the scratch path, alternating who goes first |
| `bench_ab_convert.py` | one engine binary against another in one session: `engine-cuda12/strata` and `engine-cuda12/strata.pre-change`, arms alternating, `--adapt-swaps 0 --pcie-frac 0`, `cmp` on the files each arm wrote |
| `bench_lazy_resume.py` | parked-resume A/B, three workloads (`append_only`, `older_ckpt`, `deep_rewind`): two rounds alternating which arm goes first, 2 warm-ups + 10 measured repetitions each. `--arms` names the arms, `--arm-exe arm=path` gives each its engine and `--arm-base arm=path` its config, so the same harness compares two builds on one codec (`--arms eager,lazy`) or one build on two codecs (`--arms fp32,bf16`); `--base` is the default for any arm that sets neither, `--out` names the result file. |
| `bench_capture_codec.py` | capture-and-turn latency sweep: one retained checkpoint per timed turn, at `--prompt-cache-every` 16384 and 8192, both codecs, two rounds. `STRATA_URL` must be set before `bench_lib` is imported - it reads the URL at import time. |
| `bench_memory_rerun.py` | the RAM re-measurement. Takes **every size in one invocation** and writes `memory-<codec>.head.json`, because `bench_run.py` keys its memory output on codec alone and a second size silently overwrote the first run's file - that is how the 63K bf16 run and a 32K bf16 run were lost. Measures the **first** save after a prefill separately from warm saves: the allocator admits a fixed scratch once and later saves reuse it, so a median over reps mixes two different things. `MEM_REPS` sets the repetition count (default 3; 9 was needed to separate the two). |
| `run_memory_ab.py` | drives `bench_memory_rerun.py` as a matched A/B: starts a server from `bench-bf16.json`, runs both sizes, stops it, repeats from `bench-fp32.json`, then restarts `strata-iq3_s.json` - in a `finally`, because the box may be serving someone's conversation. Kills by pid, never `pkill -f engine-cuda12/strata`, which matches the harness's own command line. |
| `bench-fp32.json` / `bench-bf16.json` | the two arm configs. **Not in the repository**: they hold this box's pack, model and tokenizer paths. Make them by copying the config you normally serve with and changing three keys - `--session-state-codec`, `log`, `slot_save_path` (the last only turns the SAVE/RESTORE HTTP commands on; set it for both arms). |
| `results/` | raw JSON per phase and arm, `REPORT.md`, `PR_COMMENT.md`, `encode_decode.txt`, `ab_engine.jsonl` |

## Running it

```sh
cd <USER>/Desktop/Strata
# arm 1: fp32
.venv/bin/python serve/server.py --engine strata --config bench-state-codec/bench-fp32.json --port 8080 &
for p in memory save_restore park_resume functional resume_paths needle; do
  (cd bench-state-codec && ../.venv/bin/python bench_run.py $p fp32 \
      --sizes 4800,31250,62500,125000 --needle-tokens 31250 --deep-tokens 62500)
done
(cd bench-state-codec && ../.venv/bin/python bench_run.py needle fp32 --needle-tokens 125000 --depths 10,50,90 --seeds 1)
(cd bench-state-codec && ../.venv/bin/python bench_cold.py fp32 31250,62500,125000 3)
# arm 2: bf16 - stop the server, restart with bench-bf16.json, same phases with `bf16`
# codec-free: encode/decode on real state bytes, and the A/B regression check
(cd bench-state-codec && ../.venv/bin/python extract_state.py ../session-bench/sr-fp32-62500.bin gdn_fp32.bin &&
  g++ -O2 -std=c++20 -I../include bench_encode.cpp ../src/core/state_codec.cpp -o bench_encode &&
  ./bench_encode gdn_fp32.bin 6 | tee results/encode_decode.txt)
.venv/bin/python tools/ab_engine.py bench-state-codec/bench-fp32.json 'A:' \
  'B:xargs=--session-state-codec;bf16' --rounds 2 --port 8090 --log bench-state-codec/results/ab_engine.jsonl
cd bench-state-codec && ../.venv/bin/python bench_report.py > results/REPORT.md
```

Each phase needs the server up and the arm's own log path exported (`STRATA_LOG=bench-bf16.log`) so the log parsers
read the right file. The `bench_ab_convert.py` comparison needs the earlier build beside the current one:

```sh
git stash push src/core/state_codec.cpp src/core/conversation_copy.hpp include/strata/core/state_codec.hpp
(cd build-cuda12 && ninja strata) && cp build-cuda12/strata engine-cuda12/strata.pre-change && git stash pop
(cd build-cuda12 && ninja strata) && cp build-cuda12/strata engine-cuda12/strata    # 63 MB each, rebuild to discard
```

Session files land in `../session-bench/` (about 13 GB after a full run; delete freely).
`--needle-tokens`/`--sizes` are character counts calibrated on this model at about 3.2 characters per token:
`31250` comes out as 31,797 tokens, `62500` as 63,026, `125000` as 121,614.

### The lazy-restoration comparison

`bench_lib.py` finds the repository from its own location; `STRATA_ROOT` overrides it. `STRATA_URL` picks the server
(default `http://127.0.0.1:8080`) and `STRATA_LOG` the engine log the parsers read.

```sh
cd <USER>/Desktop/Strata
# both builds from the same tree: HEAD, then the working tree
git stash push src/program/generate.cpp && (cd build-cuda12 && ninja strata) \
  && cp build-cuda12/strata engine-cuda12/strata.eager-2274c1c && git stash pop
(cd build-cuda12 && ninja strata) && cp build-cuda12/strata engine-cuda12/strata.lazy
(cd bench-state-codec && ../.venv/bin/python bench_lazy_resume.py \
    --sizes 31250,62500,125000 --warm 2 --reps 10 --rounds 2)      # ~45 min, one server per arm per round
# the default path on the new build: same harness, fp32 config, one arm
(cd bench-state-codec && ../.venv/bin/python bench_lazy_resume.py --base bench-state-codec/bench-fp32.json \
    --sizes 31250 --warm 1 --reps 3 --rounds 1 --arms lazy)
```

### FP32 against BF16 on one build

Same harness, both arms the same engine, the codec the only difference. This is the run that answers "is bf16 as
fast as fp32 now", so neither arm is a historical figure from another harness:

```sh
(cd bench-state-codec && ../.venv/bin/python bench_lazy_resume.py --arms fp32,bf16 \
    --arm-exe fp32=engine-cuda12/strata.lazy2 --arm-exe bf16=engine-cuda12/strata.lazy2 \
    --arm-base fp32=bench-state-codec/bench-fp32.json \
    --arm-base bf16=bench-state-codec/bench-bf16.json \
    --sizes 31250,62500,125000 --warm 2 --reps 10 --rounds 2 \
    --out lazy_resume_codec_parity.json)
```

Raw JSON in `results/lazy_resume.json` (both builds, `bf16`), `results/lazy_resume_codec_parity.json` (both codecs,
one build) and `results/lazy_resume-fp32-default.json`. The engine
lines the harness parses for the lazy arm only are `conversation cache: retained N checkpoints … (C coded, X bytes
of it)` and `resumed through the checkpoint at T tokens (…) in M ms`; `reused:` on the `prompt … = N reused + M read`
line names the checkpoint in both arms. `deep_rewind` asks a different question every repetition so no parked copy's
near-tail checkpoint matches it and the engine mounts through the periodic checkpoint at 16,384 every time.

## What the numbers mean, and two limits

* `VmRSS` does not count pages the kernel has already pushed out, and most of what Strata parks is pageable host
  memory, so the RSS columns are observed lower bounds, not high-water marks. The process's lifetime peak
  (`VmHWM`) is deliberately not reported. Once the conversation cache is at its budget, save/restore/switch
  increments go to zero because the allocator already owns the blocks - so `memory` runs on a fresh process.
* A bf16 RESTORE transient can be *larger* than fp32's at short contexts: widening coded state to the fp32 arrays
  the engine runs on allocates those arrays beside the compressed originals (that is review issue P2 in the PR).
* The server's `/v1/chat/completions` ignores `logprobs`, so no KL divergence or per-token logit comparison is
  possible through the API. Quality here is needle recall, resumed-prompt byte counts, answer text identity, and the
  codec's own error bound measured on a real state's values.