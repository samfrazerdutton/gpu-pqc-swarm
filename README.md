# GPU-Accelerated PQC Swarm Pipeline (`gpu-pqc-swarm`)

## Overview
This repository contains a high-performance, GPU-accelerated Post-Quantum Cryptography (PQC) pipeline designed for zero-latency cryptographic key exchanges in distributed drone networks. By offloading complex ML-KEM (FIPS 203) mathematical operations to NVIDIA GPUs, this architecture bypasses traditional CPU bottlenecks to enable massive scaling for autonomous swarms.

## Performance Benchmarks
*Hardware/Environment: NVIDIA GPU, WSL2 Ubuntu, Python 3.12, CUDA 12.9*

* **NTT Throughput:** ~32.3 Million NTTs/sec (8192 NTT batch)
* **1000-Node Star Topology Exchange:**
  * **KeyGen:** 6.30 ms (4.8M pairs/sec)
  * **Encapsulation:** 8.69 ms
  * **Decapsulation:** 11.25 ms
  * **Total Pipeline Execution:** 43.68 ms (0.044 ms per pair)

## Tech Stack
* **Core:** Python 3.12, CUDA C++ (PTX)
* **GPU Compute:** NVIDIA `cuQuantum` (cuTensorNet), `cuPQC`
* **JIT Compilation:** CuPy, `nvrtc`, `nvJitLink`
* **Cryptography Standards:** ML-KEM-768 (Kyber), HKDF-SHA3-256

## Architecture
Designed for topology-agnostic session management, supporting Star, Mesh, Ring, and Broadcast networks. Utilizes JIT kernel offloading to dynamically compile operations directly in VRAM, eliminating host-to-device memory transfer latency.
## Live Hardware Benchmark Output
Executing the pipeline against the custom C++ CUDA NTT kernel for a 1,000-node drone swarm routing simulation:

```text
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  DEMO 6: Scale Test — 1000 node star
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

══════════════════════════════════════════════════════════════
  PQC PIPELINE  |  1000-node star exchange
  Pairs: 1000  |  Algorithm: ML-KEM-768 (FIPS 203)
══════════════════════════════════════════════════════════════

  [ KeyGen ] 1000 responders...
  [ Encaps ] 1000 pairs...
  [ Decaps ] 1000 pairs...
  [ Derive ] 1000 session keys via HKDF-SHA3-256...

  ──────────────────────────────────────────────────────────
  KeyGen:       7.40 ms  (1000 responders)
  Encaps:       5.10 ms  (1000 pairs)
  Decaps:       3.91 ms  (1000 pairs)
  Total:       32.06 ms  (0.032 ms/pair)
  Sessions: 1000/1000 established
  Agreement:1000/1000 (✓ Verified (Constant-Time))
  ──────────────────────────────────────────────────────────
