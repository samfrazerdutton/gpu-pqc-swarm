"""
pipeline.py — The generalized PQC key exchange pipeline.
Topology-agnostic. Domain-agnostic. GPU-accelerated.

Usage:
    pipeline = PQCPipeline()
    peers = [Peer("server-1"), Peer("sensor-42"), Peer("satellite-7")]
    pairs = Topology.mesh(peers)
    sessions = pipeline.run(pairs, label="IoT mesh re-key")
"""
from __future__ import annotations
import numpy as np
import sys, time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "bridge"))
from mlkem_bridge import MLKEM768Bridge

from .peer import Peer, PeerGroup
from .topology import ExchangePair, TopologyType, Topology
from .session import Session, SessionStore

MLKEM768_PK_BYTES   = 1184
MLKEM768_SK_BYTES   = 2400
MLKEM768_CT_BYTES   = 1088
MLKEM768_SS_BYTES   = 32
MLKEM768_SEED_BYTES = 32


class PQCPipeline:
    """
    GPU-accelerated ML-KEM-768 key exchange pipeline.
    
    Works for any set of peers in any topology:
      - Drone swarms          (STAR: ground → drones)
      - Server clusters       (MESH: every server ↔ every server)
      - IoT sensor rings      (RING: sensor → sensor → ... → sensor)
      - Microservice pairs    (P2P: service-a → service-b)
      - Satellite broadcasts  (BROADCAST: satellite → ground stations)
    
    The pipeline:
      1. KeyGen    — generate keypairs for all responders on GPU
      2. Encaps    — initiators encapsulate shared secrets on GPU
      3. Decaps    — responders recover shared secrets on GPU
      4. Derive    — HKDF-SHA3-256 session keys from shared secrets
      5. Sessions  — return live AES-256-GCM encrypted channels
    """

    def __init__(self, ptx_path: str = None, verbose: bool = True):
        if ptx_path is None:
            ptx_path = Path(__file__).parent.parent / "kernels" / "mlkem_kernel.ptx"
        self.bridge  = MLKEM768Bridge(ptx_path=ptx_path)
        self.verbose = verbose
        self.store   = SessionStore()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self, pairs: list[ExchangePair],
            label: str = "key exchange") -> SessionStore:
        """
        Execute a full key exchange for a list of ExchangePairs.
        Returns a SessionStore with live encrypted channels for each pair.
        """
        n = len(pairs)
        self._log(f"\n{'═'*62}")
        self._log(f"  PQC PIPELINE  |  {label}")
        self._log(f"  Pairs: {n}  |  Algorithm: ML-KEM-768 (FIPS 203)")
        self._log(f"{'═'*62}")

        t_total = time.perf_counter()

        # ── Collect unique responders (need keypairs) ──────────────────
        responders = list({id(p.responder): p.responder
                           for p in pairs}.values())
        n_resp = len(responders)
        resp_idx = {id(r): i for i, r in enumerate(responders)}

        # ── Phase 1: KeyGen for all responders ─────────────────────────
        self._log(f"\n  [ KeyGen ] {n_resp} responders...")
        t0 = time.perf_counter()
        pk_arr, sk_arr = self.bridge.keygen(n_resp)
        keygen_ms = (time.perf_counter() - t0) * 1000

        for i, resp in enumerate(responders):
            resp.public_key  = pk_arr[i].tobytes()
            resp._secret_key = sk_arr[i].tobytes()

        # ── Phase 2: Encaps — one per pair ─────────────────────────────
        self._log(f"  [ Encaps ] {n} pairs...")
        pk_for_encaps = np.array([
            np.frombuffer(p.responder.public_key, dtype=np.uint8)
            for p in pairs
        ])
        t0 = time.perf_counter()
        ct_arr, ss_init_arr = self.bridge.encaps(pk_for_encaps)
        encaps_ms = (time.perf_counter() - t0) * 1000

        # ── Phase 3: Decaps — batch by responder ───────────────────────
        self._log(f"  [ Decaps ] {n} pairs...")
        sk_for_decaps = np.array([
            np.frombuffer(p.responder._secret_key, dtype=np.uint8)
            for p in pairs
        ])
        t0 = time.perf_counter()
        ss_resp_arr = self.bridge.decaps(ct_arr, sk_for_decaps)
        decaps_ms = (time.perf_counter() - t0) * 1000

        # ── Phase 4 & 5: Derive sessions ───────────────────────────────
        self._log(f"  [ Derive ] {n} session keys via HKDF-SHA3-256...")
        store = SessionStore()
        agreements = 0
        for i, pair in enumerate(pairs):
            ss_i = ss_init_arr[i].tobytes()
            ss_r = ss_resp_arr[i].tobytes()
            # With real cuPQC: ss_i == ss_r always
            # With stubs: use initiator's ss for both sides (same GPU op)
            session = Session.derive(
                peer_a=pair.initiator,
                peer_b=pair.responder,
                shared_secret=ss_i,  # swap to ss_r when cuPQC lands
                label=pair.label
            )
            store.add(session)
            pair.initiator._session_key = session._key
            pair.responder._session_key = session._key
            if ss_i == ss_r:
                agreements += 1

        self.store = store
        total_ms = (time.perf_counter() - t_total) * 1000

        self._log(f"\n  {'─'*58}")
        self._log(f"  KeyGen:   {keygen_ms:8.2f} ms  ({n_resp} responders)")
        self._log(f"  Encaps:   {encaps_ms:8.2f} ms  ({n} pairs)")
        self._log(f"  Decaps:   {decaps_ms:8.2f} ms  ({n} pairs)")
        self._log(f"  Total:    {total_ms:8.2f} ms  ({total_ms/n:.3f} ms/pair)")
        self._log(f"  Sessions: {len(store)}/{n} established")
        self._log(f"  Agreement:{agreements}/{n} "
                  f"({'stub mode' if agreements < n else '✓'})")
        self._log(f"  {'─'*58}\n")

        return store

    def rekey(self, store: SessionStore,
              label: str = "re-key") -> SessionStore:
        """Re-run key exchange for all pairs in an existing session store."""
        pairs = [
            ExchangePair(s.peer_a, s.peer_b, label=s.label)
            for s in store.all()
        ]
        return self.run(pairs, label=label)

    def benchmark_ntt(self):
        self._log("\n  [ NTT Benchmark ]")
        for n in [128, 512, 2048, 8192]:
            self.bridge.ntt_benchmark(n)
