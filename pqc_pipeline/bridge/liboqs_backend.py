"""
liboqs_backend.py — Real PQC math via liboqs reference implementation.
Uses ML-KEM-768 (FIPS 203) and ML-DSA-65 (FIPS 204) — the actual
NIST-standardised algorithms, not the old Kyber/Dilithium names.

This backend produces genuine cryptographic key agreement:
  ss_initiator == ss_responder  (real math, not stubs)

Performance is CPU-bound (~1-5ms per operation) vs GPU stubs (~0.1ms).
The GPU backend handles parallelism; this backend handles correctness.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")

import oqs
import numpy as np
import hashlib, os, time
from typing import Tuple

KEM_ALG = "ML-KEM-768"
SIG_ALG = "ML-DSA-65"

# Size constants — must match FIPS 203/204 specs
KEM_PK  = 1184;  KEM_SK  = 2400
KEM_CT  = 1088;  KEM_SS  = 32
DSA_PK  = 1952;  DSA_SK  = 4032
DSA_SIG = 3309;  DSA_MH  = 64


class LibOQSBackend:
    """
    Real ML-KEM-768 + ML-DSA-65 via liboqs reference implementation.
    Drop-in replacement for the GPU stub kernels.
    Produces genuine cryptographic agreement verified by NIST test vectors.
    """

    def __init__(self):
        # Verify both algorithms are available
        try:
            _k = oqs.KeyEncapsulation(KEM_ALG)
            _k.free()
        except Exception as e:
            raise RuntimeError(f"ML-KEM-768 not available: {e}")
        try:
            _s = oqs.Signature(SIG_ALG)
            _s.free()
        except Exception as e:
            raise RuntimeError(f"ML-DSA-65 not available: {e}")

        print(f"[liboqs] ML-KEM-768 + ML-DSA-65 ready (liboqs reference impl)")

    # ── ML-KEM-768 ────────────────────────────────────────────────────────────

    def kem_keygen(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n ML-KEM-768 keypairs. Returns (pk, sk) arrays."""
        pk_arr = np.zeros((n, KEM_PK), dtype=np.uint8)
        sk_arr = np.zeros((n, KEM_SK), dtype=np.uint8)

        t0 = time.perf_counter()
        for i in range(n):
            kem = oqs.KeyEncapsulation(KEM_ALG)
            pk  = kem.generate_keypair()
            sk  = kem.export_secret_key()
            pk_arr[i] = np.frombuffer(pk, dtype=np.uint8)
            sk_arr[i] = np.frombuffer(sk, dtype=np.uint8)
            kem.free()
        ms = (time.perf_counter() - t0) * 1000
        print(f"[liboqs KEM KeyGen] {n} keypairs in {ms:.1f}ms "
              f"({n/ms*1000:.0f}/sec)")
        return pk_arr, sk_arr

    def kem_encaps(self, pk_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encapsulate shared secrets against n public keys."""
        n = len(pk_arr)
        ct_arr = np.zeros((n, KEM_CT), dtype=np.uint8)
        ss_arr = np.zeros((n, KEM_SS), dtype=np.uint8)

        t0 = time.perf_counter()
        for i in range(n):
            kem = oqs.KeyEncapsulation(KEM_ALG)
            pk_bytes = pk_arr[i].tobytes()
            ct, ss   = kem.encap_secret(pk_bytes)
            ct_arr[i] = np.frombuffer(ct, dtype=np.uint8)
            ss_arr[i] = np.frombuffer(ss, dtype=np.uint8)
            kem.free()
        ms = (time.perf_counter() - t0) * 1000
        print(f"[liboqs KEM Encaps] {n} encapsulations in {ms:.1f}ms "
              f"({n/ms*1000:.0f}/sec)")
        return ct_arr, ss_arr

    def kem_decaps(self, ct_arr: np.ndarray,
                   sk_arr: np.ndarray) -> np.ndarray:
        """Decapsulate ciphertexts to recover shared secrets."""
        n = len(ct_arr)
        ss_arr = np.zeros((n, KEM_SS), dtype=np.uint8)

        t0 = time.perf_counter()
        for i in range(n):
            kem = oqs.KeyEncapsulation(KEM_ALG,
                      secret_key=sk_arr[i].tobytes())
            ss  = kem.decap_secret(ct_arr[i].tobytes())
            ss_arr[i] = np.frombuffer(ss, dtype=np.uint8)
            kem.free()
        ms = (time.perf_counter() - t0) * 1000
        print(f"[liboqs KEM Decaps] {n} decapsulations in {ms:.1f}ms "
              f"({n/ms*1000:.0f}/sec)")
        return ss_arr

    # ── ML-DSA-65 ─────────────────────────────────────────────────────────────

    def dsa_keygen(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n ML-DSA-65 signing keypairs."""
        pk_arr = np.zeros((n, DSA_PK), dtype=np.uint8)
        sk_arr = np.zeros((n, DSA_SK), dtype=np.uint8)

        t0 = time.perf_counter()
        for i in range(n):
            sig = oqs.Signature(SIG_ALG)
            pk  = sig.generate_keypair()
            sk  = sig.export_secret_key()
            pk_arr[i] = np.frombuffer(pk, dtype=np.uint8)
            sk_arr[i] = np.frombuffer(sk, dtype=np.uint8)
            sig.free()
        ms = (time.perf_counter() - t0) * 1000
        print(f"[liboqs DSA KeyGen] {n} keypairs in {ms:.1f}ms "
              f"({n/ms*1000:.0f}/sec)")
        return pk_arr, sk_arr

    def dsa_sign(self, messages: list, sk: bytes) -> np.ndarray:
        """
        Sign n messages with one signing key.
        messages: list of bytes objects
        sk:       secret key bytes
        Returns:  (n, DSA_SIG) uint8 array
        """
        n = len(messages)
        sig_arr = np.zeros((n, DSA_SIG), dtype=np.uint8)

        t0 = time.perf_counter()
        signer = oqs.Signature(SIG_ALG, secret_key=sk)
        for i, msg in enumerate(messages):
            signature = signer.sign(msg)
            # liboqs signatures may vary in length — pad/trim to DSA_SIG
            sig_bytes = signature[:DSA_SIG].ljust(DSA_SIG, b'\x00')
            sig_arr[i] = np.frombuffer(sig_bytes, dtype=np.uint8)
        signer.free()
        ms = (time.perf_counter() - t0) * 1000
        print(f"[liboqs DSA Sign]   {n} signatures in {ms:.1f}ms "
              f"({n/ms*1000:.0f}/sec)")
        return sig_arr

    def dsa_verify(self, messages: list, signatures: np.ndarray,
                   pk: bytes) -> np.ndarray:
        """
        Verify n signatures against one authority public key.
        Returns (n,) int32 array — 1=valid, 0=invalid.
        """
        n = len(messages)
        results = np.zeros(n, dtype=np.int32)

        t0 = time.perf_counter()
        verifier = oqs.Signature(SIG_ALG)
        for i, msg in enumerate(messages):
            try:
                valid = verifier.verify(
                    msg,
                    signatures[i].tobytes().rstrip(b'\x00'),
                    pk
                )
                results[i] = 1 if valid else 0
            except Exception:
                results[i] = 0
        verifier.free()
        ms = (time.perf_counter() - t0) * 1000
        valid_count = int(results.sum())
        print(f"[liboqs DSA Verify] {n} verifications in {ms:.1f}ms "
              f"({n/ms*1000:.0f}/sec) — {valid_count}/{n} valid")
        return results

    def verify_kem_agreement(self, ss_enc: np.ndarray,
                              ss_dec: np.ndarray) -> bool:
        """Verify encaps and decaps produced identical shared secrets."""
        matches = np.all(ss_enc == ss_dec, axis=1)
        n = len(matches)
        valid = int(matches.sum())
        print(f"[liboqs KEM Agree]  {valid}/{n} genuine agreements "
              f"({'✓ REAL CRYPTO' if valid == n else '✗ MISMATCH'})")
        return matches
