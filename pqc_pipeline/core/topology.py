"""
topology.py — Defines HOW peers exchange keys with each other.
The pipeline is topology-agnostic; pick the pattern that fits your network.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Iterator
from .peer import Peer, PeerGroup

class TopologyType(Enum):
    STAR      = "star"       # One hub talks to all spokes (ground→drones)
    MESH      = "mesh"       # Every peer talks to every other peer
    RING      = "ring"       # Peers form a chain: A→B→C→...→A
    P2P       = "p2p"        # Single pair exchange
    BROADCAST = "broadcast"  # One sender, N receivers (one-to-many)
    TREE      = "tree"       # Hierarchical: root→branches→leaves

@dataclass
class ExchangePair:
    """One directed key exchange: initiator encapsulates for responder."""
    initiator: Peer   # Sends ciphertext
    responder: Peer   # Holds secret key
    label: str = ""

    def __repr__(self):
        return f"{self.initiator.name} → {self.responder.name}"


class Topology:
    """
    Generates the set of ExchangePairs for a given topology pattern.
    Feed this into the Pipeline to execute the exchanges.
    """

    @staticmethod
    def star(hub: Peer, spokes: list[Peer]) -> list[ExchangePair]:
        """Hub initiates key exchange with every spoke."""
        return [ExchangePair(hub, spoke, label=f"hub→{spoke.name}")
                for spoke in spokes]

    @staticmethod
    def mesh(peers: list[Peer]) -> list[ExchangePair]:
        """Every peer establishes a session with every other peer."""
        pairs = []
        for i, a in enumerate(peers):
            for j, b in enumerate(peers):
                if i < j:
                    pairs.append(ExchangePair(a, b, label=f"{a.name}↔{b.name}"))
        return pairs

    @staticmethod
    def ring(peers: list[Peer]) -> list[ExchangePair]:
        """Each peer exchanges with the next; last wraps to first."""
        n = len(peers)
        return [ExchangePair(peers[i], peers[(i+1) % n],
                             label=f"{peers[i].name}→{peers[(i+1)%n].name}")
                for i in range(n)]

    @staticmethod
    def p2p(a: Peer, b: Peer) -> list[ExchangePair]:
        """Single directed exchange."""
        return [ExchangePair(a, b, label=f"{a.name}→{b.name}")]

    @staticmethod
    def broadcast(sender: Peer, receivers: list[Peer]) -> list[ExchangePair]:
        """One sender establishes sessions with all receivers."""
        return [ExchangePair(sender, r, label=f"broadcast→{r.name}")
                for r in receivers]

    @staticmethod
    def from_group(group: PeerGroup, topo_type: TopologyType,
                   hub: Peer = None) -> list[ExchangePair]:
        """Auto-generate pairs from a PeerGroup + topology type."""
        peers = list(group)
        if topo_type == TopologyType.STAR:
            if hub is None:
                raise ValueError("STAR topology requires a hub peer")
            spokes = [p for p in peers if p is not hub]
            return Topology.star(hub, spokes)
        elif topo_type == TopologyType.MESH:
            return Topology.mesh(peers)
        elif topo_type == TopologyType.RING:
            return Topology.ring(peers)
        elif topo_type == TopologyType.BROADCAST:
            if hub is None:
                raise ValueError("BROADCAST topology requires a sender peer")
            receivers = [p for p in peers if p is not hub]
            return Topology.broadcast(hub, receivers)
        else:
            raise ValueError(f"Use explicit Topology.{topo_type.value}()")
