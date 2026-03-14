#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================
// ML-KEM-768 Parameter Constants (FIPS 203)
// ============================================================
#define MLKEM_K          3       // Security level (768 = k*256 bits)
#define MLKEM_N          256     // Polynomial degree
#define MLKEM_Q          3329    // Prime modulus
#define MLKEM_ETA1       2       // Noise parameter eta1
#define MLKEM_ETA2       2       // Noise parameter eta2
#define MLKEM_DU         10      // Compression param du
#define MLKEM_DV         4       // Compression param dv

// Key sizes in bytes
#define MLKEM768_PUBLICKEY_BYTES   1184
#define MLKEM768_SECRETKEY_BYTES   2400
#define MLKEM768_CIPHERTEXT_BYTES  1088
#define MLKEM768_SHAREDSECRET_BYTES 32
#define MLKEM768_SYMBYTES          32

// ============================================================
// GPU-side Montgomery reduction (mod Q=3329)
// ============================================================
__device__ __forceinline__ int16_t montgomery_reduce(int32_t a) {
    // Montgomery constant: R = 2^16, R^-1 mod Q, Q^-1 mod R
    const int32_t QINV = 62209; // Q^{-1} mod 2^16
    int16_t u = (int16_t)((int16_t)a * (int16_t)QINV);
    int32_t t = (int32_t)u * MLKEM_Q;
    t = a - t;
    return (int16_t)(t >> 16);
}

// ============================================================
// Barrett reduction (faster for known-range inputs)
// ============================================================
__device__ __forceinline__ int16_t barrett_reduce(int16_t a) {
    const int32_t V = 20159; // round(2^26 / Q)
    int16_t t = (int16_t)(((int32_t)V * a + (1 << 25)) >> 26);
    return a - t * (int16_t)MLKEM_Q;
}

// ============================================================
// Number Theoretic Transform (NTT) - core of ML-KEM
// Each thread handles one of the 256 coefficients
// ============================================================
__device__ void ntt_layer(int16_t* poly, int len, int start, int16_t zeta) {
    for (int j = start; j < start + len; j++) {
        int16_t t = montgomery_reduce((int32_t)zeta * poly[j + len]);
        poly[j + len] = poly[j] - t;
        poly[j]       = poly[j] + t;
    }
}

// NTT zeta table (precomputed powers of primitive root mod Q)
__constant__ int16_t NTT_ZETAS[128] = {
    -1044, -758, -359, -1517,  1493,  1422,   287,   202,
    -171,   622,  1577,  182,   962, -1202,  -977,   800,
    1855,  -1310,  536, -1667,  -369,  -167, -1570,  1547,
    1812,  -1032,  529, -1376,   615,   638, -1523,  -330,
    -731,  -1272,  1006, -1766, -1862, -1362,   -4,  -403,
    -555,   347, -1609, -271,   -314, -1069,  1593, -1576,
    1321,  -1328,  -500,   -3,  -226,  -757,   -60,   573,
     928,  -685,   824,  -497,  -440,  -1063,  1657,  1304,
     -43,   -76,  -553,   -7,  -735,   415,  -656,   571,
    -585,  -357,  -786,  -985, -1084,  -481,  1680,  1376,
    1030,  -627,  1509,  -836,   -24,   519,   777, -1620,
      94,  -1535,  -784,  1653,   462, -1602,  1169,  -928,
    -424,  -1461,  -540,  -783,  1203,   700,  -484, -1640,
     963,   264,   -64,  -672,  1659,  -744,  1496,   980,
     -21,   -10,  -977,   -4,  1596,  -468,   888,   -20,
    -765,  1274,  1517,  -756,  -1044,   758,   359,  1517
};

__global__ void gpu_ntt_kernel(int16_t* poly_array, int num_polys) {
    int poly_idx = blockIdx.x;
    int tid      = threadIdx.x; // 0..127 (we use 128 threads per poly)

    if (poly_idx >= num_polys) return;

    int16_t* poly = poly_array + poly_idx * MLKEM_N;

    // Cooperative NTT across 128 threads
    __shared__ int16_t s_poly[MLKEM_N];

    // Load into shared memory
    s_poly[tid]       = poly[tid];
    s_poly[tid + 128] = poly[tid + 128];
    __syncthreads();

    // 7 layers of butterfly (log2(256) = 8, but layer 0 is trivial)
    int len = 128;
    int k   = 0;
    while (len >= 2) {
        int start = (tid / len) * len * 2 + (tid % len);
        int16_t zeta = NTT_ZETAS[k + tid / len];
        int16_t t = montgomery_reduce((int32_t)zeta * s_poly[start + len]);
        s_poly[start + len] = barrett_reduce(s_poly[start] - t);
        s_poly[start]       = barrett_reduce(s_poly[start] + t);
        __syncthreads();
        len >>= 1;
        k += (MLKEM_N / (len * 2));
    }

    // Write back
    poly[tid]       = s_poly[tid];
    poly[tid + 128] = s_poly[tid + 128];
}

// ============================================================
// Polynomial multiplication in NTT domain (pointwise)
// ============================================================
__global__ void gpu_poly_basemul_kernel(
    int16_t* r, const int16_t* a, const int16_t* b, int num_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs * MLKEM_N / 2) return;

    int poly_pair = idx / (MLKEM_N / 2);
    int coeff     = idx % (MLKEM_N / 2);

    const int16_t* pa = a + poly_pair * MLKEM_N + coeff * 2;
    const int16_t* pb = b + poly_pair * MLKEM_N + coeff * 2;
    int16_t*       pr = r + poly_pair * MLKEM_N + coeff * 2;

    // Precomputed zeta for basemul
    int16_t zeta = NTT_ZETAS[64 + coeff];

    // Schoolbook 2x2 negacyclic basemul
    pr[0] = (int16_t)(montgomery_reduce((int32_t)pa[0] * pb[0] +
             montgomery_reduce((int32_t)zeta * montgomery_reduce((int32_t)pa[1] * pb[1]))));
    pr[1] = (int16_t)(montgomery_reduce((int32_t)pa[0] * pb[1] +
             (int32_t)pa[1] * pb[0]));
}

// ============================================================
// Compress / Decompress for ciphertext encoding
// ============================================================
__global__ void gpu_compress_kernel(
    uint8_t* out, const int16_t* poly, int d, int n_coeffs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_coeffs) return;

    // Compress: round(2^d / Q * x) mod 2^d
    uint32_t x = ((uint32_t)(uint16_t)poly[idx] * (1u << d) + MLKEM_Q / 2) / MLKEM_Q;
    out[idx] = (uint8_t)(x & ((1u << d) - 1));
}

__global__ void gpu_decompress_kernel(
    int16_t* poly, const uint8_t* in, int d, int n_coeffs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_coeffs) return;

    // Decompress: round(Q / 2^d * x)
    poly[idx] = (int16_t)(((uint32_t)in[idx] * MLKEM_Q + (1u << (d - 1))) >> d);
}

// ============================================================
// Key Generation Kernel (Encapsulation side)
// Generates ephemeral keypair on GPU - each block = one peer
// ============================================================
__global__ void gpu_mlkem_keygen_stub(
    uint8_t* pk_out,       // [num_peers x MLKEM768_PUBLICKEY_BYTES]
    uint8_t* sk_out,       // [num_peers x MLKEM768_SECRETKEY_BYTES]
    const uint8_t* seeds,  // [num_peers x MLKEM768_SYMBYTES] random seeds
    int num_peers
) {
    int peer = blockIdx.x;
    if (peer >= num_peers) return;

    // --- In production: call cuPQC device API here ---
    // cuPQCDx::ML_KEM<cuPQCDx::ML_KEM_768>::KeyGen(
    //     pk_out + peer * MLKEM768_PUBLICKEY_BYTES,
    //     sk_out + peer * MLKEM768_SECRETKEY_BYTES,
    //     seeds  + peer * MLKEM768_SYMBYTES
    // );

    // Stub: write deterministic placeholder derived from seed
    // (Replace with real cuPQC call once headers are linked)
    uint8_t* pk = pk_out + peer * MLKEM768_PUBLICKEY_BYTES;
    uint8_t* sk = sk_out + peer * MLKEM768_SECRETKEY_BYTES;
    const uint8_t* seed = seeds + peer * MLKEM768_SYMBYTES;

    int tid = threadIdx.x;
    for (int i = tid; i < MLKEM768_PUBLICKEY_BYTES; i += blockDim.x)
        pk[i] = (uint8_t)((seed[i % MLKEM768_SYMBYTES] ^ (uint8_t)i) + peer);
    for (int i = tid; i < MLKEM768_SECRETKEY_BYTES; i += blockDim.x)
        sk[i] = (uint8_t)((seed[i % MLKEM768_SYMBYTES] ^ (uint8_t)(i >> 1)) - peer);
}

// ============================================================
// Encapsulation Kernel (sender side)
// ============================================================
__global__ void gpu_mlkem_encaps_stub(
    uint8_t* ct_out,       // [num_peers x MLKEM768_CIPHERTEXT_BYTES]
    uint8_t* ss_out,       // [num_peers x MLKEM768_SHAREDSECRET_BYTES]
    const uint8_t* pk_in,  // [num_peers x MLKEM768_PUBLICKEY_BYTES]
    const uint8_t* coins,  // [num_peers x MLKEM768_SYMBYTES] random coins
    int num_peers
) {
    int peer = blockIdx.x;
    if (peer >= num_peers) return;

    uint8_t* ct = ct_out + peer * MLKEM768_CIPHERTEXT_BYTES;
    uint8_t* ss = ss_out + peer * MLKEM768_SHAREDSECRET_BYTES;
    const uint8_t* pk   = pk_in  + peer * MLKEM768_PUBLICKEY_BYTES;
    const uint8_t* coin = coins  + peer * MLKEM768_SYMBYTES;

    int tid = threadIdx.x;
    // Stub: derive shared secret from pk + coins
    for (int i = tid; i < MLKEM768_SHAREDSECRET_BYTES; i += blockDim.x)
        ss[i] = pk[i % MLKEM768_PUBLICKEY_BYTES] ^ coin[i % MLKEM768_SYMBYTES];
    for (int i = tid; i < MLKEM768_CIPHERTEXT_BYTES; i += blockDim.x)
        ct[i] = pk[i % MLKEM768_PUBLICKEY_BYTES] ^ coin[i % MLKEM768_SYMBYTES] ^ (uint8_t)i;
}

// ============================================================
// Decapsulation Kernel (receiver side)
// ============================================================
__global__ void gpu_mlkem_decaps_stub(
    uint8_t* ss_out,       // [num_peers x MLKEM768_SHAREDSECRET_BYTES]
    const uint8_t* ct_in,  // [num_peers x MLKEM768_CIPHERTEXT_BYTES]
    const uint8_t* sk_in,  // [num_peers x MLKEM768_SECRETKEY_BYTES]
    int num_peers
) {
    int peer = blockIdx.x;
    if (peer >= num_peers) return;

    uint8_t* ss        = ss_out + peer * MLKEM768_SHAREDSECRET_BYTES;
    const uint8_t* ct  = ct_in  + peer * MLKEM768_CIPHERTEXT_BYTES;
    const uint8_t* sk  = sk_in  + peer * MLKEM768_SECRETKEY_BYTES;

    int tid = threadIdx.x;
    // Stub: recover shared secret from ct + sk
    for (int i = tid; i < MLKEM768_SHAREDSECRET_BYTES; i += blockDim.x)
        ss[i] = ct[i % MLKEM768_CIPHERTEXT_BYTES] ^ sk[i % MLKEM768_SECRETKEY_BYTES];
}
