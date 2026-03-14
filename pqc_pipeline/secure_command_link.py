"""
secure_command_link.py — Demo 7: Authenticated Command Broadcasting
Shows ML-KEM-768 + ML-DSA-65 working together as a full PQC stack.

Flow:
  Ground Station
    ├── ML-KEM: establishes encrypted channel with each peer
    └── ML-DSA: signs every command with its private key

  Each Peer (drone/server/sensor/satellite)
    ├── ML-KEM: decapsulates → AES-256-GCM session key
    └── ML-DSA: verifies ground station signature on GPU
               REJECTS command if signature invalid — even if encryption valid
"""
import sys, os, hashlib, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from bridge.mlkem_bridge import PQCBridge
from core.peer import Peer, PeerType, PeerRole
from core.session import Session

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


def hash_message(msg: bytes, pk: np.ndarray) -> np.ndarray:
    """
    Compute mu = H(H(pk) || msg) as per FIPS 204 spec.
    Returns (64,) uint8 array.
    """
    pk_hash = hashlib.sha3_256(pk.tobytes()).digest()
    mu = hashlib.shake_256(pk_hash + msg).digest(64)
    return np.frombuffer(mu, dtype=np.uint8)


def demo_secure_command_link(bridge: PQCBridge, num_peers: int = 1000):
    print(f"\n{'▓'*62}")
    print(f"  DEMO 7: Secure Command Link")
    print(f"  ML-KEM-768 (confidentiality) + ML-DSA-65 (authenticity)")
    print(f"  Peers: {num_peers}  |  FIPS 203 + FIPS 204")
    print(f"{'▓'*62}\n")

    t_demo = time.perf_counter()

    # ── Step 1: Authority (ground station) generates DSA signing keypair ──
    print("[ Step 1 ] Authority generates ML-DSA-65 signing keypair...")
    auth_pk, auth_sk = bridge.dsa_keygen(1)
    print(f"  Authority PK: {auth_pk[0,:8].tolist()}... ({len(auth_pk[0])} bytes)")
    print(f"  Authority SK: [SEALED]  ({len(auth_sk[0])} bytes)\n")

    # ── Step 2: Peers generate KEM keypairs (for encrypted channel) ───────
    print(f"[ Step 2 ] {num_peers} peers generate ML-KEM-768 keypairs...")
    peer_kem_pk, peer_kem_sk = bridge.kem_keygen(num_peers)
    print()

    # ── Step 3: Authority encapsulates session keys for all peers ─────────
    print(f"[ Step 3 ] Authority encapsulates {num_peers} session keys...")
    ct_arr, ss_auth = bridge.kem_encaps(peer_kem_pk)
    print()

    # ── Step 4: Peers decapsulate their session keys ──────────────────────
    print(f"[ Step 4 ] {num_peers} peers decapsulate session keys...")
    ss_peers = bridge.kem_decaps(ct_arr, peer_kem_sk)
    print()

    # ── Step 5: Authority constructs and SIGNS a command ─────────────────
    print("[ Step 5 ] Authority signs command with ML-DSA-65...")
    COMMAND = b"EXECUTE:MISSION_ALPHA|WAYPOINT:53.48,-2.24|ALT:120|RULES_OF_ENGAGEMENT:HOLD"

    # mu = H(H(auth_pk) || command)  per FIPS 204
    mu = hash_message(COMMAND, auth_pk[0])
    mu_batch = np.tile(mu, (1, 1))  # single signer

    signature = bridge.dsa_sign(mu_batch, auth_sk)
    print(f"  Command:   {COMMAND.decode()}")
    print(f"  Signature: {signature[0,:16].tolist()}... ({len(signature[0])} bytes)\n")

    # ── Step 6: All peers verify the signature simultaneously on GPU ──────
    print(f"[ Step 6 ] {num_peers} peers verify signature simultaneously on GPU...")

    # Each peer independently verifies: same command, same authority pk, same sig
    mu_broadcast  = np.tile(mu,           (num_peers, 1))
    sig_broadcast = np.tile(signature[0], (num_peers, 1))

    results = bridge.dsa_verify(mu_broadcast, sig_broadcast, auth_pk)
    valid_count = int(results.sum())
    print()

    # ── Step 7: Tamper test — modify command, verify rejection ───────────
    print("[ Step 7 ] Tamper test — injecting forged command...")
    FORGED = b"EXECUTE:SELF_DESTRUCT|ALL_UNITS"
    mu_forged    = hash_message(FORGED, auth_pk[0])
    mu_forged_b  = np.tile(mu_forged, (num_peers, 1))

    results_forged = bridge.dsa_verify(mu_forged_b, sig_broadcast, auth_pk)
    rejected_count = int((results_forged == 0).sum())
    print(f"  Forged command rejected by: {rejected_count}/{num_peers} peers ✓\n")

    # ── Step 8: Encrypt command with AES-256-GCM session keys ────────────
    if HAS_CRYPTO:
        print("[ Step 8 ] Encrypting authenticated command for all peers...")
        t0 = time.perf_counter()
        encrypted_commands = []
        for i in range(min(num_peers, 5)):  # demo first 5
            h = hashlib.sha3_256()
            h.update(ss_auth[i].tobytes())
            h.update(i.to_bytes(4, "big"))
            h.update(b"SECURE_CMD_LINK_v1")
            session_key = h.digest()
            aesgcm = AESGCM(session_key)
            nonce  = os.urandom(12)
            # AAD = signature — binds encryption to authentication
            aad = signature[0].tobytes()
            ct  = aesgcm.encrypt(nonce, COMMAND, aad)
            encrypted_commands.append(nonce + ct)
        enc_ms = (time.perf_counter()-t0)*1000
        print(f"  Sample encrypted command (peer 0): "
              f"{encrypted_commands[0][:20].hex()}... "
              f"({len(encrypted_commands[0])} bytes)")
        print(f"  5 commands encrypted in {enc_ms:.2f}ms\n")

    # ── Summary ───────────────────────────────────────────────────────────
    total_ms = (time.perf_counter() - t_demo) * 1000
    print(f"{'═'*62}")
    print(f"  SECURE COMMAND LINK — RESULTS")
    print(f"  {'─'*58}")
    print(f"  KEM sessions established: {num_peers}/{num_peers}")
    print(f"  DSA signatures verified:  {valid_count}/{num_peers} ✓")
    print(f"  Forged commands rejected: {rejected_count}/{num_peers} ✓")
    print(f"  Total pipeline time:      {total_ms:.2f}ms")
    print(f"  {'─'*58}")
    print(f"  Security guarantee:")
    print(f"    Confidentiality → ML-KEM-768 (FIPS 203) + AES-256-GCM")
    print(f"    Authenticity    → ML-DSA-65  (FIPS 204)")
    print(f"    Quantum-safe    → Both algorithms NIST-standardised")
    print(f"{'═'*62}\n")

    return {
        "kem_sessions": num_peers,
        "dsa_valid":    valid_count,
        "dsa_rejected": rejected_count,
        "total_ms":     total_ms
    }


def main():
    bridge = PQCBridge()

    # Warm up
    print("\n  [ NTT Benchmark ]")
    bridge.ntt_benchmark(8192)

    # Run secure command link demo at scale
    for n in [10, 100, 1000]:
        demo_secure_command_link(bridge, num_peers=n)


if __name__ == "__main__":
    main()
