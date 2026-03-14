"""
tunnel.py  —  Mutton Industries PQC Swarm Tunnel
Secure ML-KEM-768 key exchange for autonomous drone swarms.

Architecture:
    SwarmNode A (ground control) ──────── SwarmNode B (drone)
         KeyGen()                              KeyGen()
         Encaps(B.pk) → ct, ss_A              Decaps(ct, B.sk) → ss_B
         assert ss_A == ss_B  ✓
         
    Shared secret then seeds AES-256-GCM for the actual data tunnel.
"""

import numpy as np
import hashlib, os, time
from mlkem_bridge import MLKEM768Bridge

# ── AES-256-GCM symmetric layer (post-KEM) ─────────────────────────────────
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    print("[Warning] pip install cryptography  for AES-GCM tunnel layer")


class SwarmTunnel:
    """
    Full PQC tunnel for a swarm of N drones.
    
    Key Exchange Flow:
      1. Ground station runs keygen for all N drones + itself
      2. Ground encapsulates against each drone's public key  
      3. Ciphertexts broadcast to drones (public channel, safe to intercept)
      4. Each drone decapsulates → shared secret
      5. Shared secret → AES-256-GCM session key
      6. All comms encrypted with quantum-safe session keys
    """

    def __init__(self):
        self.bridge = MLKEM768Bridge()
        self._session_keys = {}  # peer_id → 32-byte AES key

    def establish_swarm_keys(self, num_drones: int):
        """
        Full key exchange ceremony for a drone swarm.
        Returns timing stats for each phase.
        """
        print(f"\n{'='*60}")
        print(f"  MUTTON INDUSTRIES — SWARM KEY EXCHANGE")
        print(f"  Drones: {num_drones}  |  Algorithm: ML-KEM-768 (FIPS 203)")
        print(f"{'='*60}\n")

        stats = {}

        # ── Phase 1: Drone keypair generation ──────────────────────────────
        print("[ Phase 1 ] Generating drone keypairs on GPU...")
        t0 = time.perf_counter()
        drone_pk, drone_sk = self.bridge.keygen(num_drones)
        stats["keygen_ms"] = (time.perf_counter() - t0) * 1000

        # ── Phase 2: Ground station encapsulation ──────────────────────────
        print("\n[ Phase 2 ] Ground station encapsulating shared secrets...")
        t0 = time.perf_counter()
        ciphertexts, ss_ground = self.bridge.encaps(drone_pk)
        stats["encaps_ms"] = (time.perf_counter() - t0) * 1000

        print(f"\n  Ciphertexts shape: {ciphertexts.shape} "
              f"({ciphertexts.nbytes / 1024:.1f} KB total broadcast)")
        print(f"  Shared secrets shape: {ss_ground.shape} (NEVER transmitted)")

        # ── Phase 3: Drone decapsulation ───────────────────────────────────
        print("\n[ Phase 3 ] Drones decapsulating on GPU...")
        t0 = time.perf_counter()
        ss_drones = self.bridge.decaps(ciphertexts, drone_sk)
        stats["decaps_ms"] = (time.perf_counter() - t0) * 1000

        # ── Phase 4: Verify agreement ──────────────────────────────────────
        print("\n[ Phase 4 ] Verifying key agreement...")
        matches = self.bridge.verify_key_agreement(ss_ground, ss_drones)

        # ── Phase 5: Derive AES-256 session keys ──────────────────────────
        print("\n[ Phase 5 ] Deriving AES-256-GCM session keys via HKDF-SHA3-256...")
        for i in range(num_drones):
            # HKDF-like derivation: SHA3-256(shared_secret || drone_id || "SWARM_TUNNEL_v1")
            h = hashlib.sha3_256()
            h.update(ss_ground[i])
            h.update(i.to_bytes(4, "big"))
            h.update(b"SWARM_TUNNEL_v1_MUTTON_INDUSTRIES")
            self._session_keys[i] = h.digest()

        # ── Summary ────────────────────────────────────────────────────────
        total_ms = sum(stats.values())
        print(f"\n{'='*60}")
        print(f"  KEY EXCHANGE COMPLETE")
        print(f"  KeyGen:  {stats['keygen_ms']:8.2f} ms")
        print(f"  Encaps:  {stats['encaps_ms']:8.2f} ms")
        print(f"  Decaps:  {stats['decaps_ms']:8.2f} ms")
        print(f"  Total:   {total_ms:8.2f} ms  ({total_ms/num_drones:.3f} ms/drone)")
        print(f"  Throughput: {num_drones/(total_ms/1000):.0f} drone-KEMs/sec")
        print(f"{'='*60}\n")

        return {
            "drone_pk":    drone_pk,
            "drone_sk":    drone_sk,
            "ciphertexts": ciphertexts,
            "ss_ground":   ss_ground,
            "ss_drones":   ss_drones,
            "matches":     matches,
            "stats":       stats
        }

    def encrypt_command(self, drone_id: int, plaintext: bytes) -> bytes:
        """Encrypt a command to a specific drone using its session key."""
        if not HAS_CRYPTO:
            raise RuntimeError("pip install cryptography")
        key = self._session_keys[drone_id]
        aesgcm = AESGCM(key)
        nonce  = os.urandom(12)
        ct     = aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ct

    def decrypt_command(self, drone_id: int, ciphertext: bytes) -> bytes:
        """Decrypt a command received from ground station."""
        if not HAS_CRYPTO:
            raise RuntimeError("pip install cryptography")
        key    = self._session_keys[drone_id]
        aesgcm = AESGCM(key)
        nonce  = ciphertext[:12]
        ct     = ciphertext[12:]
        return aesgcm.decrypt(nonce, ct, None)

    def run_ntt_benchmark(self):
        """Benchmark the raw NTT core."""
        print("\n[ NTT Benchmark ] Testing raw polynomial transform throughput...")
        for n in [128, 512, 2048, 8192]:
            self.bridge.ntt_benchmark(n)


def main():
    tunnel = SwarmTunnel()
    
    # NTT core benchmark first
    tunnel.run_ntt_benchmark()

    # Key exchange stress test across swarm sizes
    for n_drones in [10, 100, 500, 1000]:
        result = tunnel.establish_swarm_keys(n_drones)
        
        # Demo: encrypt a command to drone 0
        if HAS_CRYPTO and result["matches"][0]:
            cmd = b"WAYPOINT:53.4808,-2.2426,ALT:120,SPEED:15"
            encrypted = tunnel.encrypt_command(0, cmd)
            decrypted = tunnel.decrypt_command(0, encrypted)
            assert decrypted == cmd
            print(f"  [AES-GCM] Command tunnel verified for drone 0 ✓")
            print(f"  Plaintext:  {cmd}")
            print(f"  Ciphertext: {encrypted[:24].hex()}... ({len(encrypted)} bytes)\n")

if __name__ == "__main__":
    main()
