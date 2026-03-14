"""
test_sampling.py
Verifies CBD and SampleNTT kernels against expected properties.
CBD:       all coefficients in [-2, 2], mean ~0
SampleNTT: all coefficients in [0, 3329), uniform distribution
Matrix A:  all 9 polynomials for ML-KEM-768 valid
"""
import warnings
warnings.filterwarnings("ignore")

import sys, ctypes
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

def launch(fn, grid, block, args):
    ptrs = (ctypes.c_void_p*len(args))(*[ctypes.addressof(a) for a in args])
    _check(_cuda.cuLaunchKernel(fn,
        grid[0],grid[1],grid[2],
        block[0],block[1],block[2],
        0,None,ptrs,None))
    cp.cuda.Stream.null.synchronize()

def ptr(arr): return ctypes.c_uint64(arr.data.ptr)

ptx = Path(__file__).parent/"kernels"/"mlkem_sampling.ptx"
mod = load_ptx(ptx)
fn_cbd    = get_fn(mod, "test_cbd_kernel")
fn_ntt    = get_fn(mod, "test_samplentt_kernel")
fn_matrix = get_fn(mod, "test_gen_matrix_A")

KYBER_N = 256
KYBER_Q = 3329
KYBER_K = 3

print("\n" + "="*62)
print("  ML-KEM SAMPLING LAYER TESTS")
print("="*62)

# ── Test 1: CBD ───────────────────────────────────────────────────────────
print("\n[ Test 1 ] CBD eta=2 — noise polynomial sampling")

d_poly  = cp.zeros(KYBER_N,  dtype=cp.int16)
d_sum   = cp.zeros(1,        dtype=cp.int32)
d_max   = cp.zeros(1,        dtype=cp.int32)
d_valid = cp.zeros(1,        dtype=cp.int32)

launch(fn_cbd, (1,1,1), (1,1,1),
       [ptr(d_poly), ptr(d_sum), ptr(d_max), ptr(d_valid)])

poly  = cp.asnumpy(d_poly)
s     = int(cp.asnumpy(d_sum)[0])
m     = int(cp.asnumpy(d_max)[0])
valid = int(cp.asnumpy(d_valid)[0])

print(f"  Coefficients: {poly[:16].tolist()}...")
print(f"  Range:        [{poly.min()}, {poly.max()}]  (expected [-2, 2])")
print(f"  Sum:          {s}  (expected ~0, got {s})")
print(f"  Max |coeff|:  {m}  (must be <= 2)")
print(f"  All in range: {'✓ PASS' if valid else '✗ FAIL'}")

# distribution check
from collections import Counter
dist = Counter(poly.tolist())
print(f"  Distribution: {dict(sorted(dist.items()))}")

# ── Test 2: SampleNTT ─────────────────────────────────────────────────────
print("\n[ Test 2 ] SampleNTT — uniform polynomial from SHAKE-128")

d_poly2  = cp.zeros(KYBER_N, dtype=cp.int16)
d_valid2 = cp.zeros(1,       dtype=cp.int32)
d_min2   = cp.zeros(1,       dtype=cp.int32)
d_max2   = cp.zeros(1,       dtype=cp.int32)

launch(fn_ntt, (1,1,1), (1,1,1),
       [ptr(d_poly2), ptr(d_valid2), ptr(d_min2), ptr(d_max2)])

poly2  = cp.asnumpy(d_poly2)
valid2 = int(cp.asnumpy(d_valid2)[0])
min2   = int(cp.asnumpy(d_min2)[0])
max2   = int(cp.asnumpy(d_max2)[0])

print(f"  Coefficients: {poly2[:16].tolist()}...")
print(f"  Range:        [{min2}, {max2}]  (expected [0, {KYBER_Q-1}])")
print(f"  All in [0,Q): {'✓ PASS' if valid2 else '✗ FAIL'}")

# Uniformity check — mean should be ~Q/2
mean2 = float(poly2.mean())
print(f"  Mean:         {mean2:.1f}  (expected ~{KYBER_Q//2} for uniform)")
uniform_ok = abs(mean2 - KYBER_Q/2) < 200
print(f"  Uniformity:   {'✓ PASS' if uniform_ok else '✗ FAIL'} "
      f"(mean within 200 of Q/2)")

# ── Test 3: Full matrix A ─────────────────────────────────────────────────
print(f"\n[ Test 3 ] Matrix A generation — {KYBER_K}x{KYBER_K} polynomials")

d_A     = cp.zeros(KYBER_K*KYBER_K*KYBER_N, dtype=cp.int16)
d_valid3 = cp.zeros(KYBER_K*KYBER_K,        dtype=cp.int32)

launch(fn_matrix, (KYBER_K*KYBER_K,1,1), (256,1,1),
       [ptr(d_A), ptr(d_valid3)])

A      = cp.asnumpy(d_A).reshape(KYBER_K, KYBER_K, KYBER_N)
valid3 = cp.asnumpy(d_valid3)

all_valid = valid3.all()
print(f"  Valid polynomials: {valid3.sum()}/{KYBER_K*KYBER_K}")
for i in range(KYBER_K):
    for j in range(KYBER_K):
        v = valid3[i*KYBER_K+j]
        print(f"    A[{i}][{j}]: min={A[i,j].min():4d} "
              f"max={A[i,j].max():4d} "
              f"mean={A[i,j].mean():7.1f} "
              f"{'✓' if v else '✗'}")

# ── Check A[i][j] != A[j][i] (different seeds produce different polys)
print(f"\n  A[0][1] != A[1][0]: "
      f"{'✓ PASS' if not np.array_equal(A[0,1], A[1,0]) else '✗ FAIL'}")
print(f"  A[0][0] != A[1][1]: "
      f"{'✓ PASS' if not np.array_equal(A[0,0], A[1,1]) else '✗ FAIL'}")

# ── Summary ───────────────────────────────────────────────────────────────
cbd_pass    = bool(valid)
ntt_pass    = bool(valid2) and uniform_ok
matrix_pass = bool(all_valid)

print(f"\n{'='*62}")
print(f"  SAMPLING LAYER RESULTS")
print(f"  CBD eta=2:      {'✓ PASS' if cbd_pass    else '✗ FAIL'}")
print(f"  SampleNTT:      {'✓ PASS' if ntt_pass    else '✗ FAIL'}")
print(f"  Matrix A (3x3): {'✓ PASS' if matrix_pass else '✗ FAIL'}")

if cbd_pass and ntt_pass and matrix_pass:
    print(f"\n  ALL SAMPLING TESTS PASSED")
    print(f"  Ready to build: ByteEncode, ByteDecode, KeyGen")
else:
    print(f"\n  SOME TESTS FAILED — debug before proceeding")
print(f"{'='*62}\n")
