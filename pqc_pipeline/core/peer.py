"""
peer.py — Generic endpoint identity for the PQC pipeline.
A Peer is anything that can hold a keypair and exchange secrets:
drones, ground stations, servers, IoT sensors, satellites, browsers, etc.
"""
from __future__ import annotations
import uuid, time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class PeerRole(Enum):
    INITIATOR   = "initiator"   # Sends encapsulation (has peer's pk)
    RESPONDER   = "responder"   # Holds secret key, decapsulates
    RELAY       = "relay"       # Forwards ciphertexts, no secret access
    OBSERVER    = "observer"    # Audit/logging node

class PeerType(Enum):
    # Add any domain here — the pipeline doesn't care
    GENERIC     = "generic"
    DRONE       = "drone"
    GROUND      = "ground_station"
    SERVER      = "server"
    IOT_SENSOR  = "iot_sensor"
    SATELLITE   = "satellite"
    BROWSER     = "browser"
    MICROSERVICE= "microservice"
    HSM         = "hardware_security_module"

@dataclass
class Peer:
    """
    A generic participant in the PQC key exchange pipeline.
    Domain-agnostic: the same object models a drone, a server, or a sensor.
    """
    name: str
    peer_type: PeerType         = PeerType.GENERIC
    role: PeerRole              = PeerRole.RESPONDER
    peer_id: str                = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict              = field(default_factory=dict)

    # Set by pipeline after keygen
    public_key:  Optional[bytes] = field(default=None, repr=False)
    _secret_key: Optional[bytes] = field(default=None, repr=False)
    _session_key: Optional[bytes] = field(default=None, repr=False)
    created_at: float           = field(default_factory=time.time)

    def has_keypair(self) -> bool:
        return self.public_key is not None and self._secret_key is not None

    def has_session(self) -> bool:
        return self._session_key is not None

    def __repr__(self):
        status = "🔑" if self.has_keypair() else "○"
        session = "🔒" if self.has_session() else "○"
        return (f"Peer({self.peer_id} | {self.name} | "
                f"{self.peer_type.value} | {self.role.value} | "
                f"keypair={status} session={session})")


@dataclass
class PeerGroup:
    """
    A named collection of peers — a swarm, a cluster, a network segment.
    """
    name: str
    peers: list[Peer] = field(default_factory=list)
    group_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def add(self, peer: Peer) -> Peer:
        self.peers.append(peer)
        return peer

    def by_role(self, role: PeerRole) -> list[Peer]:
        return [p for p in self.peers if p.role == role]

    def by_type(self, peer_type: PeerType) -> list[Peer]:
        return [p for p in self.peers if p.peer_type == peer_type]

    def __len__(self):
        return len(self.peers)

    def __iter__(self):
        return iter(self.peers)

    def __repr__(self):
        return f"PeerGroup({self.name!r} | {len(self.peers)} peers)"
