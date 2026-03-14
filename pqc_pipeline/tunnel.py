"""
tunnel.py — Generalized PQC Pipeline demonstration.
Shows the same engine working across completely different domains.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.peer import Peer, PeerGroup, PeerType, PeerRole
from core.topology import Topology, TopologyType, ExchangePair
from core.pipeline import PQCPipeline
from core.session import SessionStore


def demo_drone_swarm(pipeline: PQCPipeline):
    """Original use case: ground station → drone swarm (STAR)."""
    print("\n" + "▓"*62)
    print("  DEMO 1: Drone Swarm  (STAR topology)")
    print("▓"*62)

    ground = Peer("ground-control", PeerType.GROUND, PeerRole.INITIATOR)
    drones = [Peer(f"drone-{i:03d}", PeerType.DRONE, PeerRole.RESPONDER)
              for i in range(50)]

    pairs = Topology.star(hub=ground, spokes=drones)
    store = pipeline.run(pairs, label="Drone swarm key exchange")

    # Demo: send a flight command
    session = store.get(ground, drones[0])
    if session:
        cmd = b"WAYPOINT:53.48,-2.24,ALT:120,SPEED:15"
        encrypted = session.send(cmd)
        print(f"  Flight command encrypted: {encrypted[:20].hex()}... "
              f"({len(encrypted)} bytes)")


def demo_server_cluster(pipeline: PQCPipeline):
    """Server cluster full mesh — every server trusts every other."""
    print("\n" + "▓"*62)
    print("  DEMO 2: Server Cluster  (MESH topology)")
    print("▓"*62)

    servers = [Peer(f"server-{chr(65+i)}", PeerType.SERVER, PeerRole.RESPONDER)
               for i in range(6)]

    pairs = Topology.mesh(servers)
    store = pipeline.run(pairs, label="Server cluster mesh re-key")

    print(f"  Mesh pairs: {len(pairs)} "
          f"({len(servers)} servers × {len(servers)-1} peers / 2)")
    s = store.get(servers[0], servers[1])
    if s:
        msg = b'{"action":"sync","shard":42}'
        ct  = s.send(msg)
        print(f"  RPC message encrypted: {ct[:20].hex()}... ({len(ct)} bytes)")


def demo_iot_ring(pipeline: PQCPipeline):
    """IoT sensor ring — each sensor passes keys to the next."""
    print("\n" + "▓"*62)
    print("  DEMO 3: IoT Sensor Ring  (RING topology)")
    print("▓"*62)

    sensors = [Peer(f"sensor-{i:02d}", PeerType.IOT_SENSOR, PeerRole.RESPONDER)
               for i in range(12)]

    pairs = Topology.ring(sensors)
    store = pipeline.run(pairs, label="IoT ring key rotation")

    print(f"  Ring links: {len(pairs)} sensors chained")


def demo_satellite_broadcast(pipeline: PQCPipeline):
    """Satellite broadcasts session keys to ground stations."""
    print("\n" + "▓"*62)
    print("  DEMO 4: Satellite Broadcast  (BROADCAST topology)")
    print("▓"*62)

    satellite = Peer("SAT-7", PeerType.SATELLITE, PeerRole.INITIATOR)
    ground_stations = [
        Peer(f"GS-{loc}", PeerType.GROUND, PeerRole.RESPONDER)
        for loc in ["London", "Tokyo", "NYC", "Sydney", "Dubai", "Lagos"]
    ]

    pairs = Topology.broadcast(satellite, ground_stations)
    store = pipeline.run(pairs, label="Satellite downlink key exchange")

    s = store.get(satellite, ground_stations[0])
    if s:
        telemetry = b"TELEMETRY:LAT=51.5,LON=-0.1,ALT=550km"
        ct = s.send(telemetry)
        print(f"  Telemetry encrypted: {ct[:20].hex()}... ({len(ct)} bytes)")


def demo_microservice_p2p(pipeline: PQCPipeline):
    """Two microservices establish a mutual session."""
    print("\n" + "▓"*62)
    print("  DEMO 5: Microservice P2P  (P2P topology)")
    print("▓"*62)

    auth_svc = Peer("auth-service",    PeerType.MICROSERVICE, PeerRole.INITIATOR)
    db_svc   = Peer("database-proxy", PeerType.MICROSERVICE, PeerRole.RESPONDER)

    pairs = Topology.p2p(auth_svc, db_svc)
    store = pipeline.run(pairs, label="Microservice mutual auth")

    s = store.get(auth_svc, db_svc)
    if s:
        query = b'SELECT * FROM users WHERE id=$1'
        ct = s.send(query, aad=b"db-proxy-v2")
        print(f"  DB query encrypted: {ct[:20].hex()}... ({len(ct)} bytes)")


def demo_large_scale(pipeline: PQCPipeline):
    """Stress test: 1000-node mesh key exchange."""
    print("\n" + "▓"*62)
    print("  DEMO 6: Scale Test — 1000 node star")
    print("▓"*62)

    hub   = Peer("hub", PeerType.SERVER, PeerRole.INITIATOR)
    nodes = [Peer(f"node-{i:04d}", PeerType.GENERIC, PeerRole.RESPONDER)
             for i in range(1000)]

    pairs = Topology.star(hub, nodes)
    store = pipeline.run(pairs, label="1000-node star exchange")


def main():
    pipeline = PQCPipeline(verbose=True)
    pipeline.benchmark_ntt()

    demo_drone_swarm(pipeline)
    demo_server_cluster(pipeline)
    demo_iot_ring(pipeline)
    demo_satellite_broadcast(pipeline)
    demo_microservice_p2p(pipeline)
    demo_large_scale(pipeline)

    print("\n" + "═"*62)
    print("  ALL DEMOS COMPLETE")
    print("  Same GPU pipeline. Any topology. Any domain.")
    print("═"*62 + "\n")


if __name__ == "__main__":
    main()
