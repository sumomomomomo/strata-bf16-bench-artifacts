"""Times state_encode / state_decode on the real running state of a real conversation.

The bytes come out of a session file this build wrote (the live checkpoint's `gdn` array: the GDN recurrence and
convolution state of every GDN layer, 29,417,472 fp32 values, 117,669,888 B). Nothing is invented: the values are
what the engine computed for a 63,026-token conversation on this box.

    python3 extract_state.py ../session-bench/sr-fp32-62500.bin gdn_fp32.bin
"""
from __future__ import annotations
import struct, sys

HEADER, GEOM = 64, 18


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0], o + 8


def i64(b, o):
    return struct.unpack_from("<q", b, o)[0], o + 8


def vec(b, o, elem=1):
    n, o = u64(b, o)
    return b[o:o + n * elem], o + n * elem


src, dst = sys.argv[1], sys.argv[2]
b = open(src, "rb").read()
version = struct.unpack_from("<I", b, 8)[0]
o = HEADER
for _ in range(GEOM):
    _, o = i64(b, o)
for _ in range(3):                       # layer_lo, layer_hi, cvec
    _, o = i64(b, o)
codec = 0
if version == 2:                         # a coded file says so, per checkpoint (state_codec.hpp)
    codec, o = u64(b, o)
_, o = vec(b, o, 4)                      # token ids are int32
n_img, o = u64(b, o)
o += n_img * 16                          # (start, hash) pairs
gdn, o = vec(b, o)                       # the live checkpoint's GDN state
print(f"{src}: version {version}, codec byte {codec}, gdn {len(gdn):,} B = {len(gdn)/4:,.0f} fp32 values")
if codec != 0:
    raise SystemExit("this file's state is coded; extract from an fp32 (codec fp32) file")
open(dst, "wb").write(gdn)
print(f"wrote {dst}")