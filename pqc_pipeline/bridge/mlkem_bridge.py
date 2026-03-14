import ctypes
import numpy as np
from pathlib import Path
import time

# --- CUDA Driver API Initialization ---
try:
    _cuda = ctypes.CDLL('libcuda.so')
except OSError:
    _cuda = ctypes.CDLL('libcuda.so.1')

def _check(status, msg="CUDA Error"):
    if status != 0:
        raise RuntimeError(f"{msg}: error code {status}")

# Initialize CUDA context
_cuda.cuInit(0)
device = ctypes.c_int()
_check(_cuda.cuDeviceGet(ctypes.byref(device), 0))
context = ctypes.c_void_p()
_cuda.cuCtxCreate(ctypes.byref(context), 0, device)

def _load_ptx_module(ptx_path):
    module = ctypes.c_void_p()
    with open(ptx_path, 'rb') as f:
        ptx_data = f.read() + b'\0'
    _check(_cuda.cuModuleLoadData(ctypes.byref(module), ptx_data), "cuModuleLoadData")
    return module

def _get_function(module, func_name):
    fn = ctypes.c_void_p()
    _check(_cuda.cuModuleGetFunction(ctypes.byref(fn), module, func_name.encode('utf-8')), "cuModuleGetFunction")
    return fn

class MLKEM768Bridge:
    def __init__(self, ptx_path=None):
        if ptx_path is None:
            ptx_path = Path(__file__).parent.parent / "kernels" / "mlkem_kernel.ptx"
        
        print(f"[MLKEM Bridge] Loading PTX via ctypes: {ptx_path}")
        self.module = _load_ptx_module(ptx_path)
        
        # ────────────────────────────────────────────────────────
        # This is the fix. We are loading the clean C++ names.
        # ────────────────────────────────────────────────────────
        self._keygen = _get_function(self.module, "dummy_keygen")
        self._encaps = _get_function(self.module, "dummy_encaps")
        self._decaps = _get_function(self.module, "dummy_decaps")
        self._ntt    = _get_function(self.module, "ntt_kernel")
        
        print("[MLKEM Bridge] All kernels loaded successfully.")

    def _launch(self, kernel, grid, block, *args):
        """Helper to launch CUDA kernels via ctypes."""
        arg_ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        _check(_cuda.cuLaunchKernel(
            kernel,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            0, None, arg_ptrs, None
        ), "cuLaunchKernel")
        _check(_cuda.cuCtxSynchronize(), "cuCtxSynchronize")

    def keygen(self, num_pairs):
        pk = np.zeros((num_pairs, 1184), dtype=np.uint8)
        sk = np.zeros((num_pairs, 2400), dtype=np.uint8)
        
        d_pk = ctypes.c_void_p()
        d_sk = ctypes.c_void_p()
        _cuda.cuMemAlloc(ctypes.byref(d_pk), pk.nbytes)
        _cuda.cuMemAlloc(ctypes.byref(d_sk), sk.nbytes)
        
        block = (256, 1, 1)
        grid = ((num_pairs + 255) // 256, 1, 1)
        
        c_num_pairs = ctypes.c_int(num_pairs)
        self._launch(self._keygen, grid, block, d_pk, d_sk, c_num_pairs)
        
        _cuda.cuMemcpyDtoH(pk.ctypes.data_as(ctypes.c_void_p), d_pk, pk.nbytes)
        _cuda.cuMemcpyDtoH(sk.ctypes.data_as(ctypes.c_void_p), d_sk, sk.nbytes)
        
        _cuda.cuMemFree(d_pk)
        _cuda.cuMemFree(d_sk)
        
        return pk, sk

    def encaps(self, pk_arr):
        num_pairs = len(pk_arr)
        ct = np.zeros((num_pairs, 1088), dtype=np.uint8)
        ss = np.zeros((num_pairs, 32), dtype=np.uint8)
        
        d_pk = ctypes.c_void_p()
        d_ct = ctypes.c_void_p()
        d_ss = ctypes.c_void_p()
        
        _cuda.cuMemAlloc(ctypes.byref(d_pk), pk_arr.nbytes)
        _cuda.cuMemAlloc(ctypes.byref(d_ct), ct.nbytes)
        _cuda.cuMemAlloc(ctypes.byref(d_ss), ss.nbytes)
        
        _cuda.cuMemcpyHtoD(d_pk, pk_arr.ctypes.data_as(ctypes.c_void_p), pk_arr.nbytes)
        
        block = (256, 1, 1)
        grid = ((num_pairs + 255) // 256, 1, 1)
        c_num_pairs = ctypes.c_int(num_pairs)
        
        self._launch(self._encaps, grid, block, d_pk, d_ct, d_ss, c_num_pairs)
        
        _cuda.cuMemcpyDtoH(ct.ctypes.data_as(ctypes.c_void_p), d_ct, ct.nbytes)
        _cuda.cuMemcpyDtoH(ss.ctypes.data_as(ctypes.c_void_p), d_ss, ss.nbytes)
        
        _cuda.cuMemFree(d_pk)
        _cuda.cuMemFree(d_ct)
        _cuda.cuMemFree(d_ss)
        
        return ct, ss

    def decaps(self, ct_arr, sk_arr):
        num_pairs = len(ct_arr)
        ss = np.zeros((num_pairs, 32), dtype=np.uint8)
        
        d_ct = ctypes.c_void_p()
        d_sk = ctypes.c_void_p()
        d_ss = ctypes.c_void_p()
        
        _cuda.cuMemAlloc(ctypes.byref(d_ct), ct_arr.nbytes)
        _cuda.cuMemAlloc(ctypes.byref(d_sk), sk_arr.nbytes)
        _cuda.cuMemAlloc(ctypes.byref(d_ss), ss.nbytes)
        
        _cuda.cuMemcpyHtoD(d_ct, ct_arr.ctypes.data_as(ctypes.c_void_p), ct_arr.nbytes)
        _cuda.cuMemcpyHtoD(d_sk, sk_arr.ctypes.data_as(ctypes.c_void_p), sk_arr.nbytes)
        
        block = (256, 1, 1)
        grid = ((num_pairs + 255) // 256, 1, 1)
        c_num_pairs = ctypes.c_int(num_pairs)
        
        self._launch(self._decaps, grid, block, d_ct, d_sk, d_ss, c_num_pairs)
        
        _cuda.cuMemcpyDtoH(ss.ctypes.data_as(ctypes.c_void_p), d_ss, ss.nbytes)
        
        _cuda.cuMemFree(d_ct)
        _cuda.cuMemFree(d_sk)
        _cuda.cuMemFree(d_ss)
        
        return ss

    def ntt_benchmark(self, n):
        poly = np.ones(n * 256, dtype=np.int16)
        zetas = np.ones(128, dtype=np.int16)
        
        d_poly = ctypes.c_void_p()
        d_zetas = ctypes.c_void_p()
        
        _cuda.cuMemAlloc(ctypes.byref(d_poly), poly.nbytes)
        _cuda.cuMemAlloc(ctypes.byref(d_zetas), zetas.nbytes)
        
        _cuda.cuMemcpyHtoD(d_poly, poly.ctypes.data_as(ctypes.c_void_p), poly.nbytes)
        _cuda.cuMemcpyHtoD(d_zetas, zetas.ctypes.data_as(ctypes.c_void_p), zetas.nbytes)
        
        threads_per_block = 256
        total_threads = n * 128
        blocks = (total_threads + threads_per_block - 1) // threads_per_block
        
        c_num_polys = ctypes.c_int(n)
        
        t0 = time.perf_counter()
        self._launch(self._ntt, (blocks, 1, 1), (threads_per_block, 1, 1), d_poly, d_zetas, c_num_polys)
        t_ms = (time.perf_counter() - t0) * 1000
        
        ops_sec = (n / (t_ms / 1000)) / 1e6
        print(f"[NTT Bench] {n} NTTs in {t_ms:.3f} ms ({ops_sec:.3f}M NTTs/sec)")
        
        _cuda.cuMemFree(d_poly)
        _cuda.cuMemFree(d_zetas)
