#include <stdint.h>
#include <stdio.h>

// ============================================================
// KECCAK-f[1600] permutation
// This is the core of SHA3, SHAKE-128, SHAKE-256
// All ML-KEM hashing is built on this one function
// ============================================================

#define KECCAK_ROUNDS 24

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

    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        // Theta
        for (int x = 0; x < 5; x++)
            C[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20];
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x+4)%5] ^ rotl64(C[(x+1)%5], 1);
            for (int y = 0; y < 25; y += 5)
                state[x+y] ^= D[x];
        }

        // Rho + Pi
        temp = state[1];
        for (int i = 0; i < 24; i++) {
            int j = KECCAK_PI[i];
            uint64_t t = state[j];
            state[j] = rotl64(temp, KECCAK_RHO[i]);
            temp = t;
        }

        // Chi
        for (int y = 0; y < 25; y += 5) {
            uint64_t s[5];
            for (int x = 0; x < 5; x++) s[x] = state[y+x];
            for (int x = 0; x < 5; x++)
                state[y+x] = s[x] ^ ((~s[(x+1)%5]) & s[(x+2)%5]);
        }

        // Iota
        state[0] ^= KECCAK_RC[round];
    }
}

// ── SHA3-256: used in ML-KEM for H() and G() ────────────────────────────────
__device__ void sha3_256(const uint8_t* in, size_t inlen,
                          uint8_t out[32]) {
    uint64_t state[25] = {0};
    const int rate = 136; // SHA3-256 rate = 1088 bits = 136 bytes

    // Absorb
    size_t offset = 0;
    while (inlen > 0) {
        size_t block = (inlen < (size_t)rate) ? inlen : rate;
        for (size_t i = 0; i < block; i++)
            ((uint8_t*)state)[offset + i] ^= in[i];
        in    += block;
        inlen -= block;
        offset += block;
        if (offset == (size_t)rate) {
            keccak_f1600(state);
            offset = 0;
        }
    }

    // Pad (SHA3 domain separation: 0x06)
    ((uint8_t*)state)[offset]      ^= 0x06;
    ((uint8_t*)state)[rate - 1]    ^= 0x80;
    keccak_f1600(state);

    // Squeeze
    for (int i = 0; i < 32; i++)
        out[i] = ((uint8_t*)state)[i];
}

// ── SHAKE-128: used in ML-KEM for SampleNTT (generating matrix A) ───────────
__device__ void shake128(const uint8_t* in, size_t inlen,
                          uint8_t* out, size_t outlen) {
    uint64_t state[25] = {0};
    const int rate = 168; // SHAKE-128 rate = 1344 bits = 168 bytes

    // Absorb
    size_t offset = 0;
    while (inlen > 0) {
        size_t block = (inlen < (size_t)rate) ? inlen : rate;
        for (size_t i = 0; i < block; i++)
            ((uint8_t*)state)[offset + i] ^= in[i];
        in    += block;
        inlen -= block;
        offset += block;
        if (offset == (size_t)rate) {
            keccak_f1600(state);
            offset = 0;
        }
    }

    // Pad (SHAKE domain separation: 0x1F)
    ((uint8_t*)state)[offset]   ^= 0x1F;
    ((uint8_t*)state)[rate - 1] ^= 0x80;
    keccak_f1600(state);

    // Squeeze
    while (outlen > 0) {
        size_t block = (outlen < (size_t)rate) ? outlen : rate;
        for (size_t i = 0; i < block; i++)
            out[i] = ((uint8_t*)state)[i];
        out    += block;
        outlen -= block;
        if (outlen > 0) keccak_f1600(state);
    }
}

// ── SHAKE-256: used in ML-KEM for PRF and key derivation ────────────────────
__device__ void shake256(const uint8_t* in, size_t inlen,
                          uint8_t* out, size_t outlen) {
    uint64_t state[25] = {0};
    const int rate = 136; // SHAKE-256 rate = 1088 bits = 136 bytes

    size_t offset = 0;
    while (inlen > 0) {
        size_t block = (inlen < (size_t)rate) ? inlen : rate;
        for (size_t i = 0; i < block; i++)
            ((uint8_t*)state)[offset + i] ^= in[i];
        in    += block;
        inlen -= block;
        offset += block;
        if (offset == (size_t)rate) {
            keccak_f1600(state);
            offset = 0;
        }
    }

    ((uint8_t*)state)[offset]   ^= 0x1F;
    ((uint8_t*)state)[rate - 1] ^= 0x80;
    keccak_f1600(state);

    while (outlen > 0) {
        size_t block = (outlen < (size_t)rate) ? outlen : rate;
        for (size_t i = 0; i < block; i++)
            out[i] = ((uint8_t*)state)[i];
        out    += block;
        outlen -= block;
        if (outlen > 0) keccak_f1600(state);
    }
}

// ============================================================
// TEST KERNEL — verify SHA3-256 against known answer
// Input: "abc" -> known SHA3-256 hash
// ============================================================
extern "C" {
__global__ void test_sha3_kernel(uint8_t* output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // SHA3-256("abc") known answer from NIST:
    // 3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532
    uint8_t msg[3] = {'a', 'b', 'c'};
    sha3_256(msg, 3, output);
}

__global__ void test_shake128_kernel(uint8_t* output) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // SHAKE-128("") 32 bytes known answer:
    // 7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26
    uint8_t empty[1] = {0};
    shake128(empty, 0, output, 32);
}
} // extern "C"
