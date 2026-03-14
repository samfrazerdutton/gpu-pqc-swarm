import warnings
warnings.filterwarnings("ignore")
import ctypes, sys
import numpy as np
import cupy as cp
from pathlib import Path

cp.cuda.Device(0).use()
_ = cp.zeros(1)
cp.cuda.Stream.null.synchronize()

try:
    _cuda = ctypes.CDLL('libcuda.so')
except OSError:
    _cuda = ctypes.CDLL('libcuda.so.1')

def _check(s, msg=""):
    if s != 0: raise RuntimeError(f"{msg}: error {s}")

def load_ptx(path):
    mod = ctypes.c_void_p()
    data = open(path,'rb').read() + b'\0'
    _check(_cuda.cuModuleLoadData(ctypes.byref(mod), data))
    return mod

def get_fn(mod, name):
    fn = ctypes.c_void_p()
    _check(_cuda.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()))
    return fn

def launch(fn, args):
    ptrs = (ctypes.c_void_p*len(args))(*[ctypes.addressof(a) for a in args])
    _check(_cuda.cuLaunchKernel(fn,1,1,1,1,1,1,0,None,ptrs,None))
    cp.cuda.Stream.null.synchronize()

def ptr(a): return ctypes.c_uint64(a.data.ptr)

mod = load_ptx(Path(__file__).parent/"kernels"/"mlkem_codec.ptx")
fn  = get_fn(mod, "test_codec_kernel")

d_res = cp.zeros(6, dtype=cp.int32)
launch(fn, [ptr(d_res)])
res = cp.asnumpy(d_res)

print("\n" + "="*62)
print("  ML-KEM CODEC (ByteEncode/ByteDecode) TESTS")
print("="*62)
print(f"\n  ByteEncode_12 round-trip: {'✓ PASS' if res[0] else '✗ FAIL'}")
print(f"  ByteEncode_10 round-trip: {'✓ PASS' if res[1] else '✗ FAIL'}")
print(f"  ByteEncode_4  round-trip: {'✓ PASS' if res[2] else '✗ FAIL'}")
print(f"  ByteEncode_1  round-trip: {'✓ PASS' if res[3] else '✗ FAIL'}")
print(f"\n  Compress_10 max error: {res[4]}  (bound: Q/2048 ~ 2)")
print(f"  Compress_4  max error: {res[5]}  (bound: Q/32   ~ 104)")
print(f"\n  Compress_10: {'✓ PASS' if res[4] <= 2   else '✗ FAIL'}")
print(f"  Compress_4:  {'✓ PASS' if res[5] <= 104  else '✗ FAIL'}")

all_pass = (res[0] and res[1] and res[2] and res[3]
            and res[4] <= 2 and res[5] <= 104)
print(f"\n{'='*62}")
if all_pass:
    print("  ALL CODEC TESTS PASSED")
    print("  ByteEncode/ByteDecode correct for d=1,4,10,12")
    print("  Compress/Decompress within FIPS 203 error bounds")
    print("  Ready to build: KeyGen")
else:
    print("  SOME TESTS FAILED")
print(f"{'='*62}\n")
