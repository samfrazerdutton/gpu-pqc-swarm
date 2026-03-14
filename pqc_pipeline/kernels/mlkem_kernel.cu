#include <stdint.h>

extern "C" {

// ============================================================
// Shared constants
// ============================================================
#define KYBER_N     256
#define KYBER_Q     3329
#define MLDSA_N     256
#define MLDSA_Q     8380417
#define MLDSA_L     5
#define MLDSA_SEED  32
#define MLDSA65_PK_BYTES   1952
#define MLDSA65_SK_BYTES   4032
#define MLDSA65_SIG_BYTES  3309
#define MLDSA65_MSG_HASH_BYTES 64

// ============================================================
// ML-KEM helpers
// ============================================================
__device__ int16_t reduce3329(int32_t a) {
    int32_t t = a % KYBER_Q;
    if (t < 0) t += KYBER_Q;
    return (int16_t)t;
}

// ============================================================
// ML-KEM NTT kernel
// ============================================================
__global__ void ntt_kernel(int16_t* poly, const int16_t* zetas, int num_polys) {
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int poly_idx = tid / (KYBER_N / 2);
    if (poly_idx >= num_polys) return;
    int thread_offset = tid % (KYBER_N / 2);
    int16_t zeta   = zetas[thread_offset % 128];
    int index_even = poly_idx * KYBER_N + thread_offset * 2;
    int index_odd  = index_even + 1;
    int16_t t_even = poly[index_even];
    int16_t t_odd  = poly[index_odd];
    int32_t t = reduce3329((int32_t)zeta * t_odd);
    poly[index_even] = reduce3329(t_even + t);
    poly[index_odd]  = reduce3329(t_even - t + KYBER_Q);
}

// ============================================================
// ML-KEM-768 stubs
// ============================================================
__global__ void dummy_keygen(uint8_t* pk, uint8_t* sk, int num_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    for (int i = 0; i < 32; i++) {
        pk[idx * 1184 + i] = (uint8_t)((idx + i) % 256);
        sk[idx * 2400 + i] = (uint8_t)((idx - i + 256) % 256);
    }
}

__global__ void dummy_encaps(const uint8_t* pk, uint8_t* ct, uint8_t* ss, int num_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    for (int i = 0; i < 32; i++) {
        ss[idx * 32 + i]   = pk[idx * 1184 + i] ^ 0x42;
        ct[idx * 1088 + i] = 0xAA;
    }
}

__global__ void dummy_decaps(const uint8_t* ct, const uint8_t* sk, uint8_t* ss, int num_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    for (int i = 0; i < 32; i++) {
        uint8_t pk_val = (uint8_t)((idx + i) % 256);
        ss[idx * 32 + i] = pk_val ^ 0x42;
    }
}

// ============================================================
// ML-DSA-65 helpers
// ============================================================
__device__ int32_t mldsa_reduce(int32_t a) {
    int32_t t = (int32_t)(((int64_t)a * 8265825LL) >> 46);
    return a - t * MLDSA_Q;
}

// ============================================================
// ML-DSA-65 KeyGen stub
// ============================================================
__global__ void mldsa_keygen_stub(
    uint8_t*       pk_out,
    uint8_t*       sk_out,
    const uint8_t* seeds,
    int            num_keys
) {
    int idx = blockIdx.x;
    if (idx >= num_keys) return;
    uint8_t*       pk   = pk_out + idx * MLDSA65_PK_BYTES;
    uint8_t*       sk   = sk_out + idx * MLDSA65_SK_BYTES;
    const uint8_t* seed = seeds  + idx * MLDSA_SEED;
    int tid = threadIdx.x;
    // rho shared between pk and sk (first 32 bytes) — critical for verify
    for (int i = tid; i < 32; i += blockDim.x) {
        pk[i] = seed[i];
        sk[i] = seed[i];
    }
    for (int i = tid + 32; i < MLDSA65_PK_BYTES; i += blockDim.x)
        pk[i] = (uint8_t)((seed[i % MLDSA_SEED] ^ (uint8_t)i) + idx);
    for (int i = tid + 32; i < 64; i += blockDim.x)
        sk[i] = (uint8_t)(seed[i % MLDSA_SEED] ^ 0xAA);
    for (int i = tid + 64; i < MLDSA65_SK_BYTES; i += blockDim.x)
        sk[i] = (uint8_t)((seed[i % MLDSA_SEED] + (uint8_t)(i >> 1)) ^ idx);
}

// ============================================================
// ML-DSA-65 Sign stub
// ============================================================
__global__ void mldsa_sign_stub(
    uint8_t*       sig_out,
    const uint8_t* msg_hash,
    const uint8_t* sk_in,
    int            num_signers
) {
    int idx = blockIdx.x;
    if (idx >= num_signers) return;
    uint8_t*       sig = sig_out  + idx * MLDSA65_SIG_BYTES;
    const uint8_t* mu  = msg_hash + idx * MLDSA65_MSG_HASH_BYTES;
    const uint8_t* sk  = sk_in    + idx * MLDSA65_SK_BYTES;
    int tid = threadIdx.x;
    // c_tilde: first 32 bytes = mu XOR rho (sk[0..31] == rho == pk[0..31])
    for (int i = tid; i < 32; i += blockDim.x)
        sig[i] = mu[i % MLDSA65_MSG_HASH_BYTES] ^ sk[i % MLDSA_SEED];
    // z vector
    for (int i = tid + 32; i < MLDSA65_SIG_BYTES; i += blockDim.x)
        sig[i] = (uint8_t)(mu[i % MLDSA65_MSG_HASH_BYTES] +
                            sk[i % MLDSA_SEED] +
                            (uint8_t)(i >> 2));
}

// ============================================================
// ML-DSA-65 Verify stub
// ============================================================
__global__ void mldsa_verify_stub(
    int*           results,
    const uint8_t* msg_hash,
    const uint8_t* sig_in,
    const uint8_t* pk_in,
    int            num_verifiers
) {
    int idx = blockIdx.x;
    if (idx >= num_verifiers) return;
    const uint8_t* mu  = msg_hash + idx * MLDSA65_MSG_HASH_BYTES;
    const uint8_t* sig = sig_in   + idx * MLDSA65_SIG_BYTES;
    const uint8_t* pk  = pk_in    + idx * MLDSA65_PK_BYTES;
    int tid = threadIdx.x;
    __shared__ int s_invalid;
    if (tid == 0) s_invalid = 0;
    __syncthreads();
    // Verify c_tilde = mu XOR rho (pk[0..31] == rho)
    for (int i = tid; i < 32; i += blockDim.x) {
        uint8_t expected = mu[i % MLDSA65_MSG_HASH_BYTES] ^ pk[i % MLDSA_SEED];
        if (sig[i] != expected)
            atomicOr(&s_invalid, 1);
    }
    __syncthreads();
    if (tid == 0)
        results[idx] = (s_invalid == 0) ? 1 : 0;
}

} // extern "C"
