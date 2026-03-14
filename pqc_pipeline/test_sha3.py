"""
test_sha3.py
Verifies our CUDA SHA3-256 and SHAKE-128 implementations
against NIST known answer test vectors.
If this passes, the hashing layer is correct and we can
build SampleNTT and CBD on top of it.
"""
import warnings
warnings.filterwarnings("ignore")

import sys, ctypes
import numpy as np
import cupy as cp
from pathlib import Path

# Init CuPy
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
    _check(_cuda.cuModuleLoadData(ctypes.byref(mod), data), "load")
    return mod

def get_fn(mod, name):
    fn = ctypes.c_void_p()
    _check(_cuda.cuModuleGetFunction(ctypes.byref(fn), mod,
           name.encode()), f"get {name}")
    return fn

def launch(fn, args):
    ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    _check(_cuda.cuLaunchKernel(fn,1,1,1,1,1,1,0,None,ptrs,None), "launch")
    cp.cuda.Stream.null.synchronize()

ptx_path = Path(__file__).parent / "kernels" / "sha3_test.ptx"
mod = load_ptx(ptx_path)
fn_sha3   = get_fn(mod, "test_sha3_kernel")
fn_shake  = get_fn(mod, "test_shake128_kernel")

print("\n" + "="*62)
print("  SHA3 / SHAKE NIST KNOWN ANSWER TESTS")
print("="*62)

# ── Test 1: SHA3-256("abc") ───────────────────────────────────────────────
EXPECTED_SHA3 = bytes.fromhex(
    "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
)
out = cp.zeros(32, dtype=cp.uint8)
ptr = ctypes.c_uint64(out.data.ptr)
launch(fn_sha3, [ptr])
result = cp.asnumpy(out).tobytes()

print(f"\n  SHA3-256('abc')")
print(f"  Expected: {EXPECTED_SHA3.hex()}")
print(f"  Got:      {result.hex()}")
print(f"  Result:   {'✓ PASS' if result == EXPECTED_SHA3 else '✗ FAIL'}")

# ── Test 2: SHAKE-128("") ─────────────────────────────────────────────────
EXPECTED_SHAKE = bytes.fromhex(
    "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26"
)
out2 = cp.zeros(32, dtype=cp.uint8)
ptr2 = ctypes.c_uint64(out2.data.ptr)
launch(fn_shake, [ptr2])
result2 = cp.asnumpy(out2).tobytes()

print(f"\n  SHAKE-128('')  32 bytes")
print(f"  Expected: {EXPECTED_SHAKE.hex()}")
print(f"  Got:      {result2.hex()}")
print(f"  Result:   {'✓ PASS' if result2 == EXPECTED_SHAKE else '✗ FAIL'}")

sha3_ok  = result  == EXPECTED_SHA3
shake_ok = result2 == EXPECTED_SHAKE

print(f"\n{'='*62}")
if sha3_ok and shake_ok:
    print("  ALL HASH TESTS PASSED")
    print("  Keccak-f[1600] implementation is correct.")
    print("  Ready to build: SampleNTT, CBD, ByteEncode")
else:
    print("  HASH TESTS FAILED — debug keccak_f1600 before proceeding")
print(f"{'='*62}\n")
