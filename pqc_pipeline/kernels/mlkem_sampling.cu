#include <stdint.h>

// ============================================================
// ML-KEM Sampling Layer
// Depends on: sha3_test.cu (keccak_f1600, shake128, shake256)
// ============================================================

#define KYBER_N    256
#define KYBER_Q    3329
#define KYBER_K    3      // ML-KEM-768
#define KYBER_ETA1 2      // noise parameter for s, e
#define KYBER_ETA2 2      // noise parameter for r, e1, e2

// ── Paste the full Keccak/SHA3/SHAKE layer here so this file is self-contained
// (copy from sha3_test.cu)

__constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

__constant__ int KECCAK_RHO[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

__constant__ int KECCAK_PI[24] = {
    10,  7, 11, 17, 18, 3, 5, 16,  8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22,  9,  6, 1
};

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ void keccak_f1600(uint64_t state[25]) {
    uint64_t C[5], D[5], temp;
    for (int round = 0; round < 24; round++) {
        for (int x = 0; x < 5; x++)
            C[x] = state[x]^state[x+5]^state[x+10]^state[x+15]^state[x+20];
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x+4)%5] ^ rotl64(C[(x+1)%5], 1);
            for (int y = 0; y < 25; y += 5) state[x+y] ^= D[x];
        }
        temp = state[1];
        for (int i = 0; i < 24; i++) {
            int j = KECCAK_PI[i];
            uint64_t t = state[j];
            state[j] = rotl64(temp, KECCAK_RHO[i]);
            temp = t;
        }
        for (int y = 0; y < 25; y += 5) {
            uint64_t s[5];
            for (int x = 0; x < 5; x++) s[x] = state[y+x];
            for (int x = 0; x < 5; x++)
                state[y+x] = s[x] ^ ((~s[(x+1)%5]) & s[(x+2)%5]);
        }
        state[0] ^= KECCAK_RC[round];
    }
}

__device__ void shake128(const uint8_t* in, size_t inlen,
                          uint8_t* out, size_t outlen) {
    uint64_t state[25] = {0};
    const int rate = 168;
    size_t offset = 0;
    while (inlen > 0) {
        size_t block = (inlen < (size_t)rate) ? inlen : rate;
        for (size_t i = 0; i < block; i++)
            ((uint8_t*)state)[offset+i] ^= in[i];
        in += block; inlen -= block; offset += block;
        if (offset == (size_t)rate) { keccak_f1600(state); offset = 0; }
    }
    ((uint8_t*)state)[offset]   ^= 0x1F;
    ((uint8_t*)state)[rate-1]   ^= 0x80;
    keccak_f1600(state);
    while (outlen > 0) {
        size_t block = (outlen < (size_t)rate) ? outlen : rate;
        for (size_t i = 0; i < block; i++) out[i] = ((uint8_t*)state)[i];
        out += block; outlen -= block;
        if (outlen > 0) keccak_f1600(state);
    }
}

__device__ void shake256(const uint8_t* in, size_t inlen,
                          uint8_t* out, size_t outlen) {
    uint64_t state[25] = {0};
    const int rate = 136;
    size_t offset = 0;
    while (inlen > 0) {
        size_t block = (inlen < (size_t)rate) ? inlen : rate;
        for (size_t i = 0; i < block; i++)
            ((uint8_t*)state)[offset+i] ^= in[i];
        in += block; inlen -= block; offset += block;
        if (offset == (size_t)rate) { keccak_f1600(state); offset = 0; }
    }
    ((uint8_t*)state)[offset]   ^= 0x1F;
    ((uint8_t*)state)[rate-1]   ^= 0x80;
    keccak_f1600(state);
    while (outlen > 0) {
        size_t block = (outlen < (size_t)rate) ? outlen : rate;
        for (size_t i = 0; i < block; i++) out[i] = ((uint8_t*)state)[i];
        out += block; outlen -= block;
        if (outlen > 0) keccak_f1600(state);
    }
}

__device__ void sha3_256(const uint8_t* in, size_t inlen, uint8_t out[32]) {
    uint64_t state[25] = {0};
    const int rate = 136;
    size_t offset = 0;
    while (inlen > 0) {
        size_t block = (inlen < (size_t)rate) ? inlen : rate;
        for (size_t i = 0; i < block; i++)
            ((uint8_t*)state)[offset+i] ^= in[i];
        in += block; inlen -= block; offset += block;
        if (offset == (size_t)rate) { keccak_f1600(state); offset = 0; }
    }
    ((uint8_t*)state)[offset]   ^= 0x06;
    ((uint8_t*)state)[rate-1]   ^= 0x80;
    keccak_f1600(state);
    for (int i = 0; i < 32; i++) out[i] = ((uint8_t*)state)[i];
}

__device__ void sha3_512(const uint8_t* in, size_t inlen, uint8_t out[64]) {
    uint64_t state[25] = {0};
    const int rate = 72; // SHA3-512 rate = 576 bits = 72 bytes
    size_t offset = 0;
    while (inlen > 0) {
        size_t block = (inlen < (size_t)rate) ? inlen : rate;
        for (size_t i = 0; i < block; i++)
            ((uint8_t*)state)[offset+i] ^= in[i];
        in += block; inlen -= block; offset += block;
        if (offset == (size_t)rate) { keccak_f1600(state); offset = 0; }
    }
    ((uint8_t*)state)[offset]   ^= 0x06;
    ((uint8_t*)state)[rate-1]   ^= 0x80;
    keccak_f1600(state);
    for (int i = 0; i < 64; i++) out[i] = ((uint8_t*)state)[i];
}

// ============================================================
// ML-KEM PRF: PRF(s, b) = SHAKE-256(s || b, 64 bytes)
// Used to generate noise polynomial seeds
// s = 32-byte seed, b = 1-byte counter
// ============================================================
__device__ void mlkem_prf(const uint8_t s[32], uint8_t b,
                           uint8_t out[64]) {
    uint8_t input[33];
    for (int i = 0; i < 32; i++) input[i] = s[i];
    input[32] = b;
    shake256(input, 33, out, 64);
}

// ============================================================
// ML-KEM G function: G(seed) = SHA3-512(seed) -> (rho, sigma)
// Used in KeyGen to expand the 32-byte seed into:
//   rho  (32 bytes) — public seed for matrix A
//   sigma (32 bytes) — private seed for secret/noise polynomials
// ============================================================
__device__ void mlkem_G(const uint8_t* seed, size_t seedlen,
                         uint8_t rho[32], uint8_t sigma[32]) {
    uint8_t digest[64];
    sha3_512(seed, seedlen, digest);
    for (int i = 0; i < 32; i++) rho[i]   = digest[i];
    for (int i = 0; i < 32; i++) sigma[i] = digest[i+32];
}

// ============================================================
// ML-KEM H function: H(input) = SHA3-256(input)
// Used to hash the public key
// ============================================================
__device__ void mlkem_H(const uint8_t* input, size_t len,
                         uint8_t out[32]) {
    sha3_256(input, len, out);
}

// ============================================================
// CBD (Centered Binomial Distribution) — FIPS 203 Algorithm 7
//
// Samples a polynomial with coefficients in [-eta, +eta]
// by computing sum of eta pairs of random bits.
//
// For ML-KEM-768: eta=2
//   Each coefficient = (b0+b1) - (b2+b3)  where bi are random bits
//   Range: [-2, +2]
//
// This is how ML-KEM generates "small" secret and noise polynomials.
// The smallness is what makes the scheme secure and correct.
// ============================================================
__device__ void cbd_eta2(const uint8_t prf_output[64],
                          int16_t poly[KYBER_N]) {
    // Each byte gives us 4 pairs of bits -> 4 coefficients
    // 64 bytes * 4 coefficients/byte = 256 coefficients = KYBER_N ✓
    for (int i = 0; i < KYBER_N / 4; i++) {
        uint8_t byte_a = prf_output[2*i];
        uint8_t byte_b = prf_output[2*i + 1];

        // Extract 8 bits from each byte, pair them up
        for (int j = 0; j < 4; j++) {
            // Each pair: a_bits - b_bits where each is sum of 2 bits
            int a = ((byte_a >> (2*j)) & 1) + ((byte_a >> (2*j+1)) & 1);
            int b = ((byte_b >> (2*j)) & 1) + ((byte_b >> (2*j+1)) & 1);
            poly[4*i + j] = (int16_t)(a - b);
        }
    }
}

// ============================================================
// SampleNTT — FIPS 203 Algorithm 6
//
// Generates one polynomial of matrix A from (rho, i, j)
// Uses SHAKE-128 as an XOF (extendable output function)
// Rejection samples until 256 coefficients in [0, Q) are found
//
// This is called K*K times to build the full K×K matrix A
// For ML-KEM-768: 3×3 = 9 calls
// ============================================================
__device__ void sample_ntt(const uint8_t rho[32], uint8_t i, uint8_t j,
                             int16_t poly[KYBER_N]) {
    // Input to SHAKE-128: rho || j || i  (note: j first per FIPS 203)
    uint8_t seed[34];
    for (int k = 0; k < 32; k++) seed[k] = rho[k];
    seed[32] = j;
    seed[33] = i;

    // Generate a stream of pseudorandom bytes via SHAKE-128
    // We need more than 256*2 bytes due to rejection sampling
    // 504 bytes gives ~99.9% chance of getting all 256 coefficients
    uint8_t stream[504];
    shake128(seed, 34, stream, 504);

    // Rejection sampling: take 3 bytes at a time, parse two 12-bit values
    // Accept if value < Q=3329, reject otherwise
    int coeff_idx = 0;
    int byte_idx  = 0;

    while (coeff_idx < KYBER_N && byte_idx + 2 < 504) {
        // Parse two 12-bit values from 3 bytes
        uint16_t d1 = ((uint16_t)stream[byte_idx] |
                       ((uint16_t)(stream[byte_idx+1] & 0x0F) << 8));
        uint16_t d2 = (((uint16_t)stream[byte_idx+1] >> 4) |
                       ((uint16_t)stream[byte_idx+2] << 4));
        byte_idx += 3;

        if (d1 < KYBER_Q && coeff_idx < KYBER_N)
            poly[coeff_idx++] = (int16_t)d1;
        if (d2 < KYBER_Q && coeff_idx < KYBER_N)
            poly[coeff_idx++] = (int16_t)d2;
    }
}

// ============================================================
// TEST KERNELS — verify CBD and SampleNTT
// ============================================================

extern "C" {

// Test CBD: sample noise polynomial, verify coefficients in [-2, 2]
// and verify distribution looks correct (mean ~0)
__global__ void test_cbd_kernel(
    int16_t* poly_out,    // [256] output polynomial
    int32_t* sum_out,     // sum of all coefficients (should be ~0)
    int32_t* max_out,     // max absolute value (should be <= 2)
    int32_t* valid_out    // 1 if all coefficients in [-2,2], else 0
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Fixed seed for reproducibility
    uint8_t seed[32];
    for (int i = 0; i < 32; i++) seed[i] = (uint8_t)(i * 7 + 13);

    // Generate PRF output
    uint8_t prf_out[64];
    mlkem_prf(seed, 0, prf_out);

    // Sample polynomial
    int16_t poly[KYBER_N];
    cbd_eta2(prf_out, poly);

    // Copy to output
    for (int i = 0; i < KYBER_N; i++) poly_out[i] = poly[i];

    // Verify: all coefficients in [-2, 2]
    int32_t sum = 0, max_abs = 0, valid = 1;
    for (int i = 0; i < KYBER_N; i++) {
        int32_t v = poly[i];
        sum += v;
        int32_t av = v < 0 ? -v : v;
        if (av > max_abs) max_abs = av;
        if (av > 2) valid = 0;
    }
    *sum_out   = sum;
    *max_out   = max_abs;
    *valid_out = valid;
}

// Test SampleNTT: verify uniform distribution in [0, Q)
// Uses fixed rho, checks all 256 coefficients are valid
__global__ void test_samplentt_kernel(
    int16_t* poly_out,    // [256] output polynomial
    int32_t* valid_out,   // 1 if all coefficients in [0, Q)
    int32_t* min_out,     // minimum coefficient
    int32_t* max_out      // maximum coefficient
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Use a fixed rho (as if from KeyGen)
    uint8_t rho[32];
    for (int i = 0; i < 32; i++) rho[i] = (uint8_t)(i + 1);

    // Sample A[0][0]
    int16_t poly[KYBER_N];
    sample_ntt(rho, 0, 0, poly);

    for (int i = 0; i < KYBER_N; i++) poly_out[i] = poly[i];

    int32_t valid = 1, min_v = 9999, max_v = -1;
    for (int i = 0; i < KYBER_N; i++) {
        int32_t v = poly[i];
        if (v < 0 || v >= KYBER_Q) valid = 0;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    *valid_out = valid;
    *min_out   = min_v;
    *max_out   = max_v;
}

// Test full matrix A generation (K*K = 9 polynomials for ML-KEM-768)
// Each block handles one polynomial A[i][j]
__global__ void test_gen_matrix_A(
    int16_t* A_out,       // [K*K*N] = [9*256] output matrix
    int32_t* valid_out    // [9] validity flags
) {
    int block_id = blockIdx.x;  // 0..8
    if (block_id >= KYBER_K * KYBER_K) return;

    uint8_t i = block_id / KYBER_K;  // row
    uint8_t j = block_id % KYBER_K;  // col

    // Fixed rho
    uint8_t rho[32];
    for (int k = 0; k < 32; k++) rho[k] = (uint8_t)(k + 1);

    int16_t poly[KYBER_N];
    sample_ntt(rho, i, j, poly);

    // Write to output
    int16_t* dest = A_out + block_id * KYBER_N;
    int tid = threadIdx.x;
    for (int k = tid; k < KYBER_N; k += blockDim.x)
        dest[k] = poly[k];

    // Validate (thread 0 only)
    if (tid == 0) {
        int32_t valid = 1;
        for (int k = 0; k < KYBER_N; k++)
            if (poly[k] < 0 || poly[k] >= KYBER_Q) { valid = 0; break; }
        valid_out[block_id] = valid;
    }
}

} // extern "C"
