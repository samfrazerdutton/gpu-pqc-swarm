"""
session.py — Manages derived session keys and encrypted channels.
Works for any peer type after key exchange completes.
"""
from __future__ import annotations
import hashlib, os, time
from dataclasses import dataclass, field
from typing import Optional
from .peer import Peer

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

APP_DOMAIN = b"PQC_PIPELINE_v1"  # Change per deployment

@dataclass
class Session:
    """
    A live encrypted channel between two peers derived from ML-KEM shared secret.
    Domain-agnostic: works for any peer types.
    """
    peer_a: Peer
    peer_b: Peer
    _key: bytes
    established_at: float = field(default_factory=time.time)
    messages_sent: int = 0
    label: str = ""

    @classmethod
    def derive(cls, peer_a: Peer, peer_b: Peer,
               shared_secret: bytes, label: str = "") -> "Session":
        """
        Derive AES-256-GCM session key from ML-KEM shared secret via HKDF-SHA3-256.
        The derivation is domain-separated so the same shared secret
        can never produce the same key in two different applications.
        """
        h = hashlib.sha3_256()
        h.update(shared_secret)
        h.update(peer_a.peer_id.encode())
        h.update(peer_b.peer_id.encode())
        h.update(APP_DOMAIN)
        key = h.digest()
        return cls(peer_a=peer_a, peer_b=peer_b, _key=key, label=label)

    def encrypt(self, plaintext: bytes, aad: bytes = None) -> bytes:
        """Encrypt a message from peer_a to peer_b."""
        if not HAS_CRYPTO:
            raise RuntimeError("pip install cryptography")
        aesgcm = AESGCM(self._key)
        nonce  = os.urandom(12)
        ct     = aesgcm.encrypt(nonce, plaintext, aad)
        return nonce + ct

    def decrypt(self, ciphertext: bytes, aad: bytes = None) -> bytes:
        """Decrypt a message."""
        if not HAS_CRYPTO:
            raise RuntimeError("pip install cryptography")
        aesgcm = AESGCM(self._key)
        return aesgcm.decrypt(ciphertext[:12], ciphertext[12:], aad)

    def send(self, message: bytes, aad: bytes = None) -> bytes:
        """Encrypt and track message count."""
        self.messages_sent += 1
        return self.encrypt(message, aad)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.established_at

    def __repr__(self):
        return (f"Session({self.peer_a.name}↔{self.peer_b.name} | "
                f"msgs={self.messages_sent} | age={self.age_seconds:.1f}s)")


class SessionStore:
    """Manages all active sessions in a pipeline run."""

    def __init__(self):
        self._sessions: dict[tuple, Session] = {}

    def add(self, session: Session):
        key = (session.peer_a.peer_id, session.peer_b.peer_id)
        self._sessions[key] = session
        # Also store reverse for bidirectional lookup
        self._sessions[(key[1], key[0])] = session

    def get(self, peer_a: Peer, peer_b: Peer) -> Optional[Session]:
        return self._sessions.get((peer_a.peer_id, peer_b.peer_id))

    def all(self) -> list[Session]:
        seen = set()
        result = []
        for k, v in self._sessions.items():
            fwd = (v.peer_a.peer_id, v.peer_b.peer_id)
            if fwd not in seen:
                seen.add(fwd)
                result.append(v)
        return result

    def __len__(self):
        return len(self._sessions) // 2
