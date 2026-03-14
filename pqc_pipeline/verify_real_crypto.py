"""
verify_real_crypto.py
Proves genuine ML-KEM-768 + ML-DSA-65 cryptographic correctness
using the liboqs reference implementation.

This is the production-honest validation:
  - Real key generation (not deterministic stubs)
  - Real encapsulation/decapsulation round-trip
  - ss_initiator == ss_responder verified cryptographically
  - Real signing and verification
  - Tamper detection with genuine signature rejection
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="oqs")

import sys, hashlib, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from bridge.liboqs_backend import LibOQSBackend
import numpy as np

def run_verification():
    print("\n" + "═"*62)
    print("  REAL CRYPTOGRAPHY VERIFICATION")
    print("  ML-KEM-768 (FIPS 203) + ML-DSA-65 (FIPS 204)")
    print("  Backend: liboqs reference implementation")
    print("═"*62 + "\n")

    backend = LibOQSBackend()

    # ── Test 1: KEM round-trip agreement ─────────────────────────────────
    print("\n[ Test 1 ] ML-KEM-768 genuine key agreement")
    print("  Generating 10 real keypairs...")
    pk_arr, sk_arr = backend.kem_keygen(10)

    print("  Encapsulating 10 shared secrets...")
    ct_arr, ss_enc = backend.kem_encaps(pk_arr)

    print("  Decapsulating 10 ciphertexts...")
    ss_dec = backend.kem_decaps(ct_arr, sk_arr)

    matches = backend.verify_kem_agreement(ss_enc, ss_dec)
    assert matches.all(), "KEM agreement failed!"

    print(f"\n  Sample shared secret (peer 0):")
    print(f"  Initiator: {ss_enc[0].tolist()[:8]}...")
    print(f"  Responder: {ss_dec[0].tolist()[:8]}...")
    print(f"  Match: {'✓ IDENTICAL' if np.array_equal(ss_enc[0], ss_dec[0]) else '✗ DIFFERENT'}")

    # ── Test 2: DSA sign and verify ───────────────────────────────────────
    print("\n[ Test 2 ] ML-DSA-65 genuine sign and verify")
    print("  Generating authority keypair...")
    auth_pk_arr, auth_sk_arr = backend.dsa_keygen(1)
    auth_pk = auth_pk_arr[0].tobytes()
    auth_sk = auth_sk_arr[0].tobytes()

    COMMAND = b"EXECUTE:MISSION_ALPHA|WAYPOINT:53.48,-2.24|ALT:120"
    print(f"  Signing: {COMMAND.decode()}")
    sigs = backend.dsa_sign([COMMAND], auth_sk)

    print("  Verifying signature...")
    results = backend.dsa_verify([COMMAND], sigs, auth_pk)
    assert results[0] == 1, "Signature verification failed!"
    print(f"  Result: ✓ GENUINE SIGNATURE VERIFIED")

    # ── Test 3: Tamper detection ──────────────────────────────────────────
    print("\n[ Test 3 ] Tamper detection with real crypto")
    FORGED = b"EXECUTE:SELF_DESTRUCT|ALL_UNITS"
    print(f"  Verifying forged command: {FORGED.decode()}")
    forged_results = backend.dsa_verify([FORGED], sigs, auth_pk)
    assert forged_results[0] == 0, "Forged command was not rejected!"
    print(f"  Result: ✓ FORGED COMMAND REJECTED BY REAL CRYPTO")

    # ── Test 4: Scale test ────────────────────────────────────────────────
    print("\n[ Test 4 ] Scale test — 50 peer genuine KEM")
    pk50, sk50   = backend.kem_keygen(50)
    ct50, ss_e50 = backend.kem_encaps(pk50)
    ss_d50       = backend.kem_decaps(ct50, sk50)
    matches50    = backend.verify_kem_agreement(ss_e50, ss_d50)
    assert matches50.all(), "Scale test failed!"

    # ── Test 5: Full pipeline with real shared secrets ────────────────────
    print("\n[ Test 5 ] AES-256-GCM session from real shared secret")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    h = hashlib.sha3_256()
    h.update(ss_enc[0].tobytes())
    h.update(b"VERIFY_TEST_v1")
    session_key = h.digest()

    aesgcm  = AESGCM(session_key)
    nonce   = os.urandom(12)
    payload = b"Quantum-safe message from genuine ML-KEM-768 shared secret"
    ct      = aesgcm.encrypt(nonce, payload, None)
    pt      = aesgcm.decrypt(nonce, ct, None)
    assert pt == payload
    print(f"  Session key: {session_key.hex()[:32]}...")
    print(f"  Encrypted:   {ct[:20].hex()}...")
    print(f"  Decrypted:   {pt.decode()}")
    print(f"  Result: ✓ AES-256-GCM SESSION FROM REAL ML-KEM-768 SECRET")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "═"*62)
    print("  VERIFICATION COMPLETE — ALL TESTS PASSED")
    print("  ─"*30)
    print("  ✓ ML-KEM-768 genuine key agreement (10/10 peers)")
    print("  ✓ ML-KEM-768 scale test (50/50 peers)")
    print("  ✓ ML-DSA-65 genuine signature verified")
    print("  ✓ ML-DSA-65 forged command rejected")
    print("  ✓ AES-256-GCM session from real shared secret")
    print("  ─"*30)
    print("  This is production-honest cryptography.")
    print("  Same algorithms. Same key sizes. NIST FIPS 203 + 204.")
    print("═"*62 + "\n")

if __name__ == "__main__":
    run_verification()
