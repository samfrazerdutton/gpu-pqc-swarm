"""
mlkem_bridge.py
Loads PTX via ctypes -> libcuda.so directly, bypassing broken CuPy driver wrappers.
"""

import ctypes, ctypes.util
import cupy as cp
import numpy as np
import os, time
from pathlib import Path

# ── CUDA driver via ctypes ───────────────────────────────────────────────────
_cuda = ctypes.CDLL("libcuda.so.1")

def _check(status, msg="CUDA driver error"):
    if status != 0:
        raise RuntimeError(f"{msg}: error code {status}")

def _init_cuda():
    _check(_cuda.cuInit(0), "cuInit failed")
    device = ctypes.c_int()
    _check(_cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet failed")
    context = ctypes.c_void_p()
    _check(_cuda.cuCtxCreate(ctypes.byref(context), 0, device), "cuCtxCreate failed")
    return context

def _load_ptx_module(ptx_path: str):
    """Load PTX into CuPy existing context via cuModuleLoad."""
    # Force CuPy to init its context FIRST, then piggyback on it
    import cupy as cp
    cp.cuda.Device(0).use()
    _ = cp.zeros(1)  # triggers full CuPy context init
    cp.cuda.Stream.null.synchronize()
    module = ctypes.c_void_p()
    path_bytes = str(ptx_path).encode("utf-8")
    _check(_cuda.cuModuleLoad(ctypes.byref(module), path_bytes),
           "cuModuleLoad failed")
    return module

def _get_function(module, name: str):
    """Get a kernel function handle from a module."""
    fn = ctypes.c_void_p()
    _check(_cuda.cuModuleGetFunction(ctypes.byref(fn), module,
                                      name.encode("utf-8")),
           f"cuModuleGetFunction({name}) failed")
    return fn

def _launch_kernel(fn, grid, block, args_list):
    """Launch a CUDA kernel with pointer-based args array."""
    # Build array of pointers to each argument
    c_args = []
    for a in args_list:
        if isinstance(a, np.int32):
            c_val = ctypes.c_int(int(a))
        elif isinstance(a, int):
            c_val = ctypes.c_size_t(a)
        else:
            c_val = ctypes.c_size_t(int(a))
        c_args.append(ctypes.addressof(c_val) if False else c_val)

    # Pack as void* array
    void_p_arr = (ctypes.c_void_p * len(args_list))()
    c_vals = []
    for i, a in enumerate(args_list):
        if isinstance(a, np.int32):
            v = ctypes.c_int(int(a))
        else:
            v = ctypes.c_uint64(int(a))
        c_vals.append(v)
        void_p_arr[i] = ctypes.cast(ctypes.byref(v), ctypes.c_void_p)

    stream = cp.cuda.Stream.null.ptr

    status = _cuda.cuLaunchKernel(
        fn,
        grid[0], grid[1], grid[2],
        block[0], block[1], block[2],
        0,
        ctypes.c_void_p(stream),
        void_p_arr,
        None
    )
    _check(status, "cuLaunchKernel failed")
    cp.cuda.Stream.null.synchronize()

# ── Constants ────────────────────────────────────────────────────────────────
MLKEM768_PK_BYTES   = 1184
MLKEM768_SK_BYTES   = 2400
MLKEM768_CT_BYTES   = 1088
MLKEM768_SS_BYTES   = 32
MLKEM768_SEED_BYTES = 32
THREADS_PER_BLOCK   = 256

class MLKEM768Bridge:
    def __init__(self, ptx_path=None):
        if ptx_path is None:
            ptx_path = Path(__file__).parent / "kernels" / "mlkem_kernel.ptx"

        print(f"[MLKEM Bridge] Loading PTX via ctypes: {ptx_path}")
        self.module   = _load_ptx_module(ptx_path)
        self._keygen  = _get_function(self.module, "_Z21gpu_mlkem_keygen_stubPhS_PKhi")
        self._encaps  = _get_function(self.module, "_Z21gpu_mlkem_encaps_stubPhS_PKhS1_i")
        self._decaps  = _get_function(self.module, "_Z21gpu_mlkem_decaps_stubPhPKhS1_i")
        self._ntt     = _get_function(self.module, "_Z14gpu_ntt_kernelPsi")
        print("[MLKEM Bridge] All kernels loaded successfully.")

    def keygen(self, num_peers: int, seeds: np.ndarray = None):
        if seeds is None:
            seeds = np.frombuffer(
                os.urandom(num_peers * MLKEM768_SEED_BYTES), dtype=np.uint8
            ).reshape(num_peers, MLKEM768_SEED_BYTES)

        d_seeds = cp.asarray(seeds)
        d_pk    = cp.zeros((num_peers, MLKEM768_PK_BYTES), dtype=cp.uint8)
        d_sk    = cp.zeros((num_peers, MLKEM768_SK_BYTES),  dtype=cp.uint8)

        t0 = time.perf_counter()
        _launch_kernel(self._keygen,
            grid=(num_peers, 1, 1), block=(THREADS_PER_BLOCK, 1, 1),
            args_list=[d_pk.data.ptr, d_sk.data.ptr, d_seeds.data.ptr,
                       np.int32(num_peers)])
        elapsed = time.perf_counter() - t0
        print(f"[KeyGen] {num_peers} keypairs in {elapsed*1000:.3f} ms "
              f"({num_peers/elapsed:.0f} keypairs/sec)")
        return cp.asnumpy(d_pk), cp.asnumpy(d_sk)

    def encaps(self, pk: np.ndarray, coins: np.ndarray = None):
        num_peers = pk.shape[0]
        if coins is None:
            coins = np.frombuffer(
                os.urandom(num_peers * MLKEM768_SEED_BYTES), dtype=np.uint8
            ).reshape(num_peers, MLKEM768_SEED_BYTES)

        d_pk    = cp.asarray(pk)
        d_coins = cp.asarray(coins)
        d_ct    = cp.zeros((num_peers, MLKEM768_CT_BYTES), dtype=cp.uint8)
        d_ss    = cp.zeros((num_peers, MLKEM768_SS_BYTES), dtype=cp.uint8)

        t0 = time.perf_counter()
        _launch_kernel(self._encaps,
            grid=(num_peers, 1, 1), block=(THREADS_PER_BLOCK, 1, 1),
            args_list=[d_ct.data.ptr, d_ss.data.ptr, d_pk.data.ptr,
                       d_coins.data.ptr, np.int32(num_peers)])
        elapsed = time.perf_counter() - t0
        print(f"[Encaps] {num_peers} encapsulations in {elapsed*1000:.3f} ms "
              f"({num_peers/elapsed:.0f} ops/sec)")
        return cp.asnumpy(d_ct), cp.asnumpy(d_ss)

    def decaps(self, ct: np.ndarray, sk: np.ndarray):
        num_peers = ct.shape[0]
        d_ct = cp.asarray(ct)
        d_sk = cp.asarray(sk)
        d_ss = cp.zeros((num_peers, MLKEM768_SS_BYTES), dtype=cp.uint8)

        t0 = time.perf_counter()
        _launch_kernel(self._decaps,
            grid=(num_peers, 1, 1), block=(THREADS_PER_BLOCK, 1, 1),
            args_list=[d_ss.data.ptr, d_ct.data.ptr, d_sk.data.ptr,
                       np.int32(num_peers)])
        elapsed = time.perf_counter() - t0
        print(f"[Decaps] {num_peers} decapsulations in {elapsed*1000:.3f} ms "
              f"({num_peers/elapsed:.0f} ops/sec)")
        return cp.asnumpy(d_ss)

    def verify_key_agreement(self, ss_sender: np.ndarray, ss_receiver: np.ndarray):
        matches = np.all(ss_sender == ss_receiver, axis=1)
        print(f"[Verify] Key agreement: {matches.sum()}/{len(matches)} "
              f"({matches.mean()*100:.1f}%)")
        return matches

    def ntt_benchmark(self, num_polys: int = 1024):
        d_polys = cp.random.randint(-1664, 1664,
                                    size=(num_polys, 256), dtype=cp.int16)
        t0 = time.perf_counter()
        _launch_kernel(self._ntt,
            grid=(num_polys, 1, 1), block=(128, 1, 1),
            args_list=[d_polys.data.ptr, np.int32(num_polys)])
        elapsed = time.perf_counter() - t0
        print(f"[NTT Bench] {num_polys} NTTs in {elapsed*1000:.3f} ms "
              f"({num_polys/elapsed/1e6:.3f}M NTTs/sec)")
        return elapsed
