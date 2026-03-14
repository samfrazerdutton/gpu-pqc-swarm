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
