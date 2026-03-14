"""
pqc_bridge.py — ML-KEM-768 + ML-DSA-65 GPU bridge
Uses CuPy for memory management, ctypes for kernel dispatch.
"""
import ctypes, os, time
import numpy as np
import cupy as cp
from pathlib import Path

# ── Init CuPy context first ──────────────────────────────────────────────────
cp.cuda.Device(0).use()
_ = cp.zeros(1)
cp.cuda.Stream.null.synchronize()

# ── CUDA driver for kernel launch only ──────────────────────────────────────
try:
    _cuda = ctypes.CDLL('libcuda.so')
except OSError:
    _cuda = ctypes.CDLL('libcuda.so.1')

def _check(status, msg="CUDA Error"):
    if status != 0:
        raise RuntimeError(f"{msg}: error code {status}")

def _load_ptx(ptx_path):
    module = ctypes.c_void_p()
    with open(ptx_path, 'rb') as f:
        data = f.read() + b'\0'
    _check(_cuda.cuModuleLoadData(ctypes.byref(module), data), "cuModuleLoadData")
    return module

def _get_fn(module, name):
    fn = ctypes.c_void_p()
    _check(_cuda.cuModuleGetFunction(
        ctypes.byref(fn), module, name.encode()), f"cuModuleGetFunction({name})")
    return fn

def _launch(kernel, grid, block, args):
    """Launch kernel — all args must be ctypes values or CuPy device pointers."""
    ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    _check(_cuda.cuLaunchKernel(
        kernel,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        0, None, ptrs, None), "cuLaunchKernel")
    cp.cuda.Stream.null.synchronize()

def _ptr(cupy_arr):
    """Get device pointer as ctypes value from CuPy array."""
    return ctypes.c_uint64(cupy_arr.data.ptr)

def _int(n):
    return ctypes.c_int(n)

# ── Size constants ────────────────────────────────────────────────────────────
KEM_PK=1184; KEM_SK=2400; KEM_CT=1088; KEM_SS=32
DSA_PK=1952; DSA_SK=4032; DSA_SIG=3309; DSA_SEED=32; DSA_MH=64
BLOCK=256

class PQCBridge:
    def __init__(self, ptx_path=None):
        if ptx_path is None:
            ptx_path = Path(__file__).parent.parent / "kernels" / "mlkem_kernel.ptx"
        print(f"[PQC Bridge] Loading PTX: {ptx_path}")
        self.module      = _load_ptx(ptx_path)
        self._kem_keygen = _get_fn(self.module, "dummy_keygen")
        self._kem_encaps = _get_fn(self.module, "dummy_encaps")
        self._kem_decaps = _get_fn(self.module, "dummy_decaps")
        self._kem_ntt    = _get_fn(self.module, "ntt_kernel")
        self._dsa_keygen = _get_fn(self.module, "mldsa_keygen_stub")
        self._dsa_sign   = _get_fn(self.module, "mldsa_sign_stub")
        self._dsa_verify = _get_fn(self.module, "mldsa_verify_stub")
        print("[PQC Bridge] ML-KEM-768 + ML-DSA-65 kernels loaded.")

    # ── ML-KEM-768 ────────────────────────────────────────────────────────────

    def kem_keygen(self, n):
        d_pk = cp.zeros((n, KEM_PK), dtype=cp.uint8)
        d_sk = cp.zeros((n, KEM_SK), dtype=cp.uint8)
        t0 = time.perf_counter()
        _launch(self._kem_keygen,
                ((n+BLOCK-1)//BLOCK,1,1), (BLOCK,1,1),
                [_ptr(d_pk), _ptr(d_sk), _int(n)])
        ms = (time.perf_counter()-t0)*1000
        pk, sk = cp.asnumpy(d_pk), cp.asnumpy(d_sk)
        print(f"[KEM KeyGen] {n} pairs in {ms:.3f}ms ({n/ms*1000:.0f}/sec)")
        return pk, sk

    def kem_encaps(self, pk_arr):
        n = len(pk_arr)
        d_pk = cp.asarray(pk_arr)
        d_ct = cp.zeros((n, KEM_CT), dtype=cp.uint8)
        d_ss = cp.zeros((n, KEM_SS), dtype=cp.uint8)
        t0 = time.perf_counter()
        _launch(self._kem_encaps,
                ((n+BLOCK-1)//BLOCK,1,1), (BLOCK,1,1),
                [_ptr(d_pk), _ptr(d_ct), _ptr(d_ss), _int(n)])
        ms = (time.perf_counter()-t0)*1000
        ct, ss = cp.asnumpy(d_ct), cp.asnumpy(d_ss)
        print(f"[KEM Encaps] {n} pairs in {ms:.3f}ms ({n/ms*1000:.0f}/sec)")
        return ct, ss

    def kem_decaps(self, ct_arr, sk_arr):
        n = len(ct_arr)
        d_ct = cp.asarray(ct_arr)
        d_sk = cp.asarray(sk_arr)
        d_ss = cp.zeros((n, KEM_SS), dtype=cp.uint8)
        t0 = time.perf_counter()
        _launch(self._kem_decaps,
                ((n+BLOCK-1)//BLOCK,1,1), (BLOCK,1,1),
                [_ptr(d_ct), _ptr(d_sk), _ptr(d_ss), _int(n)])
        ms = (time.perf_counter()-t0)*1000
        ss = cp.asnumpy(d_ss)
        print(f"[KEM Decaps] {n} pairs in {ms:.3f}ms ({n/ms*1000:.0f}/sec)")
        return ss

    # ── ML-DSA-65 ─────────────────────────────────────────────────────────────

    def dsa_keygen(self, n, seeds=None):
        if seeds is None:
            seeds = np.frombuffer(
                os.urandom(n * DSA_SEED), dtype=np.uint8).reshape(n, DSA_SEED)
        d_seeds = cp.asarray(seeds)
        d_pk    = cp.zeros((n, DSA_PK), dtype=cp.uint8)
        d_sk    = cp.zeros((n, DSA_SK), dtype=cp.uint8)
        t0 = time.perf_counter()
        _launch(self._dsa_keygen,
                (n,1,1), (BLOCK,1,1),
                [_ptr(d_pk), _ptr(d_sk), _ptr(d_seeds), _int(n)])
        ms = (time.perf_counter()-t0)*1000
        pk, sk = cp.asnumpy(d_pk), cp.asnumpy(d_sk)
        print(f"[DSA KeyGen] {n} keypairs in {ms:.3f}ms ({n/ms*1000:.0f}/sec)")
        return pk, sk

    def dsa_sign(self, msg_hashes, sk_arr):
        n = len(msg_hashes)
        d_mh  = cp.asarray(msg_hashes)
        d_sk  = cp.asarray(sk_arr)
        d_sig = cp.zeros((n, DSA_SIG), dtype=cp.uint8)
        t0 = time.perf_counter()
        _launch(self._dsa_sign,
                (n,1,1), (BLOCK,1,1),
                [_ptr(d_sig), _ptr(d_mh), _ptr(d_sk), _int(n)])
        ms = (time.perf_counter()-t0)*1000
        sig = cp.asnumpy(d_sig)
        print(f"[DSA Sign]   {n} signatures in {ms:.3f}ms ({n/ms*1000:.0f}/sec)")
        return sig

    def dsa_verify(self, msg_hashes, signatures, authority_pk):
        n = len(msg_hashes)
        # Broadcast authority pk across all verifiers
        if authority_pk.ndim == 1:
            pk_batch = np.tile(authority_pk, (n, 1))
        else:
            pk_batch = np.tile(authority_pk[0], (n, 1))
        d_mh  = cp.asarray(msg_hashes)
        d_sig = cp.asarray(signatures)
        d_pk  = cp.asarray(pk_batch)
        d_res = cp.zeros(n, dtype=cp.int32)
        t0 = time.perf_counter()
        _launch(self._dsa_verify,
                (n,1,1), (BLOCK,1,1),
                [_ptr(d_res), _ptr(d_mh), _ptr(d_sig), _ptr(d_pk), _int(n)])
        ms = (time.perf_counter()-t0)*1000
        results = cp.asnumpy(d_res)
        valid = int(results.sum())
        print(f"[DSA Verify] {n} verifications in {ms:.3f}ms "
              f"({n/ms*1000:.0f}/sec) — {valid}/{n} valid")
        return results

    def ntt_benchmark(self, n):
        poly  = cp.ones((n, 256), dtype=cp.int16)
        zetas = cp.ones(128, dtype=cp.int16)
        total = n * 128
        blocks = (total + BLOCK - 1) // BLOCK
        t0 = time.perf_counter()
        _launch(self._kem_ntt,
                (blocks,1,1), (BLOCK,1,1),
                [_ptr(poly), _ptr(zetas), _int(n)])
        ms = (time.perf_counter()-t0)*1000
        print(f"[NTT KEM]    {n} NTTs in {ms:.3f}ms ({n/ms/1000:.3f}M/sec)")

# Backwards-compatible alias
MLKEM768Bridge = PQCBridge
