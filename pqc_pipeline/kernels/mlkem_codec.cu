#include <stdint.h>

// ============================================================
// ML-KEM ByteEncode / ByteDecode — FIPS 203 Algorithms 4 & 5
//
// These functions serialise polynomial coefficients into bytes
// and deserialise them back. They are used to pack:
//   - Public key:  t (encoded as 12-bit values) + rho
//   - Secret key:  s (encoded as 12-bit values)
//   - Ciphertext:  u (10-bit) + v (4-bit compressed)
//
// ByteEncode_d: encodes N coefficients using d bits each
// ByteDecode_d: decodes bytes back into N coefficients
//
// Key insight: coefficients are NOT stored as 16-bit integers
// in the public key/ciphertext. They are bit-packed to save space.
// This is why ML-KEM-768 public key is 1184 bytes, not 256*2*3=1536.
// ============================================================

#define KYBER_N 256
#define KYBER_Q 3329
#define KYBER_K 3

// ============================================================
// ByteEncode_12: pack 256 coefficients into 256*12/8 = 384 bytes
// Used for public key polynomial t and secret key s
// Each coefficient is in [0, 2^12) = [0, 4096)
// ============================================================
__device__ void byte_encode_12(const int16_t poly[KYBER_N],
                                 uint8_t out[384]) {
    for (int i = 0; i < KYBER_N / 2; i++) {
        uint16_t a = (uint16_t)poly[2*i];
        uint16_t b = (uint16_t)poly[2*i + 1];
        // Pack two 12-bit values into 3 bytes
        out[3*i]     = (uint8_t)(a & 0xFF);
        out[3*i + 1] = (uint8_t)((a >> 8) | ((b & 0x0F) << 4));
        out[3*i + 2] = (uint8_t)(b >> 4);
    }
}

// ============================================================
// ByteDecode_12: unpack 384 bytes into 256 coefficients
// ============================================================
__device__ void byte_decode_12(const uint8_t in[384],
                                 int16_t poly[KYBER_N]) {
    for (int i = 0; i < KYBER_N / 2; i++) {
        uint8_t b0 = in[3*i];
        uint8_t b1 = in[3*i + 1];
        uint8_t b2 = in[3*i + 2];
        poly[2*i]     = (int16_t)((b0 | ((uint16_t)(b1 & 0x0F) << 8)));
        poly[2*i + 1] = (int16_t)(((b1 >> 4) | ((uint16_t)b2 << 4)));
    }
}

// ============================================================
// ByteEncode_10: pack 256 coefficients into 256*10/8 = 320 bytes
// Used for ciphertext u vector (compressed with du=10)
// ============================================================
__device__ void byte_encode_10(const int16_t poly[KYBER_N],
                                 uint8_t out[320]) {
    for (int i = 0; i < KYBER_N / 4; i++) {
        uint16_t a = (uint16_t)poly[4*i]     & 0x3FF;
        uint16_t b = (uint16_t)poly[4*i + 1] & 0x3FF;
        uint16_t c = (uint16_t)poly[4*i + 2] & 0x3FF;
        uint16_t d = (uint16_t)poly[4*i + 3] & 0x3FF;
        // Pack four 10-bit values into 5 bytes
        out[5*i]     = (uint8_t)(a);
        out[5*i + 1] = (uint8_t)((a >> 8) | (b << 2));
        out[5*i + 2] = (uint8_t)((b >> 6) | (c << 4));
        out[5*i + 3] = (uint8_t)((c >> 4) | (d << 6));
        out[5*i + 4] = (uint8_t)(d >> 2);
    }
}

// ============================================================
// ByteDecode_10: unpack 320 bytes into 256 coefficients
// ============================================================
__device__ void byte_decode_10(const uint8_t in[320],
                                 int16_t poly[KYBER_N]) {
    for (int i = 0; i < KYBER_N / 4; i++) {
        uint8_t b0 = in[5*i];
        uint8_t b1 = in[5*i + 1];
        uint8_t b2 = in[5*i + 2];
        uint8_t b3 = in[5*i + 3];
        uint8_t b4 = in[5*i + 4];
        poly[4*i]     = (int16_t)((b0 | ((uint16_t)(b1 & 0x03) << 8)) & 0x3FF);
        poly[4*i + 1] = (int16_t)(((b1 >> 2) | ((uint16_t)(b2 & 0x0F) << 6)) & 0x3FF);
        poly[4*i + 2] = (int16_t)(((b2 >> 4) | ((uint16_t)(b3 & 0x3F) << 4)) & 0x3FF);
        poly[4*i + 3] = (int16_t)(((b3 >> 6) | ((uint16_t)b4 << 2)) & 0x3FF);
    }
}

// ============================================================
// ByteEncode_4: pack 256 coefficients into 256*4/8 = 128 bytes
// Used for ciphertext v (compressed with dv=4)
// ============================================================
__device__ void byte_encode_4(const int16_t poly[KYBER_N],
                                uint8_t out[128]) {
    for (int i = 0; i < KYBER_N / 2; i++) {
        uint8_t a = (uint8_t)(poly[2*i]     & 0x0F);
        uint8_t b = (uint8_t)(poly[2*i + 1] & 0x0F);
        out[i] = a | (b << 4);
    }
}

// ============================================================
// ByteDecode_4: unpack 128 bytes into 256 coefficients
// ============================================================
__device__ void byte_decode_4(const uint8_t in[128],
                                int16_t poly[KYBER_N]) {
    for (int i = 0; i < KYBER_N / 2; i++) {
        poly[2*i]     = (int16_t)(in[i] & 0x0F);
        poly[2*i + 1] = (int16_t)(in[i] >> 4);
    }
}

// ============================================================
// ByteEncode_1: pack 256 binary coefficients into 32 bytes
// Used for message encoding (1 bit per coefficient)
// ============================================================
__device__ void byte_encode_1(const int16_t poly[KYBER_N],
                                uint8_t out[32]) {
    for (int i = 0; i < 32; i++) {
        uint8_t byte = 0;
        for (int j = 0; j < 8; j++)
            byte |= (uint8_t)((poly[8*i + j] & 1) << j);
        out[i] = byte;
    }
}

// ============================================================
// ByteDecode_1: unpack 32 bytes into 256 binary coefficients
// ============================================================
__device__ void byte_decode_1(const uint8_t in[32],
                                int16_t poly[KYBER_N]) {
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 8; j++)
            poly[8*i + j] = (int16_t)((in[i] >> j) & 1);
}

// ============================================================
// Compress_d: reduce coefficient mod Q then compress to d bits
// Compress(x, d) = round(2^d / Q * x) mod 2^d
// ============================================================
__device__ __forceinline__ int16_t compress(int16_t x, int d) {
    uint32_t mask = (1u << d) - 1;
    uint32_t val  = (uint32_t)(uint16_t)x;
    // round(2^d * x / Q) = floor((2^d * x + Q/2) / Q)
    return (int16_t)(((val * (1u << d) + KYBER_Q / 2) / KYBER_Q) & mask);
}

// ============================================================
// Decompress_d: expand d-bit value back to [0, Q)
// Decompress(y, d) = round(Q / 2^d * y)
// ============================================================
__device__ __forceinline__ int16_t decompress(int16_t y, int d) {
    return (int16_t)(((uint32_t)(uint16_t)y * KYBER_Q + (1u << (d-1))) >> d);
}

// ============================================================
// Compress/decompress full polynomial
// ============================================================
__device__ void compress_poly(const int16_t in[KYBER_N],
                                int16_t out[KYBER_N], int d) {
    for (int i = 0; i < KYBER_N; i++)
        out[i] = compress(in[i], d);
}

__device__ void decompress_poly(const int16_t in[KYBER_N],
                                  int16_t out[KYBER_N], int d) {
    for (int i = 0; i < KYBER_N; i++)
        out[i] = decompress(in[i], d);
}

// ============================================================
// NTT — needed for matrix-vector multiply (included here
// so codec.cu is self-contained for the matrix mul test)
// ============================================================
__device__ __forceinline__ int16_t montgomery_reduce(int32_t a) {
    const int32_t QINV = 62209;
    int16_t u = (int16_t)((int16_t)a * (int16_t)QINV);
    int32_t t = (int32_t)u * KYBER_Q;
    return (int16_t)((a - t) >> 16);
}

__device__ __forceinline__ int16_t barrett_reduce(int16_t a) {
    const int32_t V = 20159;
    int16_t t = (int16_t)(((int32_t)V * a + (1 << 25)) >> 26);
    return a - t * (int16_t)KYBER_Q;
}

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

__device__ void ntt(int16_t poly[KYBER_N]) {
    int len = 128, k = 0;
    while (len >= 2) {
        for (int start = 0; start < KYBER_N; start += 2*len) {
            int16_t zeta = NTT_ZETAS[k++];
            for (int j = start; j < start + len; j++) {
                int16_t t = montgomery_reduce((int32_t)zeta * poly[j+len]);
                poly[j+len] = barrett_reduce(poly[j] - t);
                poly[j]     = barrett_reduce(poly[j] + t);
            }
        }
        len >>= 1;
    }
}

__device__ void inv_ntt(int16_t poly[KYBER_N]) {
    const int16_t F = 1441; // 128^{-1} mod 3329
    int len = 2, k = 127;
    while (len <= 128) {
        for (int start = 0; start < KYBER_N; start += 2*len) {
            int16_t zeta = -NTT_ZETAS[k--];
            for (int j = start; j < start + len; j++) {
                int16_t t   = poly[j];
                poly[j]     = barrett_reduce(t + poly[j+len]);
                poly[j+len] = montgomery_reduce((int32_t)zeta * (int32_t)(poly[j+len] - t));
            }
        }
        len <<= 1;
    }
    for (int i = 0; i < KYBER_N; i++)
        poly[i] = montgomery_reduce((int32_t)F * poly[i]);
}

// ============================================================
// Pointwise multiplication in NTT domain
// ============================================================
__device__ void poly_basemul_ntt(const int16_t a[KYBER_N],
                                   const int16_t b[KYBER_N],
                                   int16_t r[KYBER_N]) {
    for (int i = 0; i < KYBER_N / 4; i++) {
        int16_t zeta = NTT_ZETAS[64 + i];
        // Basemul for degree-2 factors
        r[4*i]   = montgomery_reduce(
            (int32_t)a[4*i+1]*b[4*i+1]);
        r[4*i]   = montgomery_reduce(
            (int32_t)zeta * r[4*i]);
        r[4*i]   = montgomery_reduce(
            (int32_t)a[4*i]*b[4*i] + r[4*i]);
        r[4*i+1] = montgomery_reduce(
            (int32_t)a[4*i]*b[4*i+1] +
            (int32_t)a[4*i+1]*b[4*i]);
        // Second pair
        int16_t zeta2 = -zeta;
        r[4*i+2] = montgomery_reduce(
            (int32_t)a[4*i+3]*b[4*i+3]);
        r[4*i+2] = montgomery_reduce(
            (int32_t)zeta2 * r[4*i+2]);
        r[4*i+2] = montgomery_reduce(
            (int32_t)a[4*i+2]*b[4*i+2] + r[4*i+2]);
        r[4*i+3] = montgomery_reduce(
            (int32_t)a[4*i+2]*b[4*i+3] +
            (int32_t)a[4*i+3]*b[4*i+2]);
    }
}

// ============================================================
// TEST KERNELS
// ============================================================
extern "C" {

// Test ByteEncode/ByteDecode round-trip for all bit widths
__global__ void test_codec_kernel(int32_t* results) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int16_t poly[KYBER_N];
    int16_t poly2[KYBER_N];

    // ── Test 12-bit round-trip ────────────────────────────────────────
    // Fill with values in [0, Q)
    for (int i = 0; i < KYBER_N; i++)
        poly[i] = (int16_t)((i * 13 + 7) % KYBER_Q);

    uint8_t buf12[384];
    byte_encode_12(poly, buf12);
    byte_decode_12(buf12, poly2);

    int ok12 = 1;
    for (int i = 0; i < KYBER_N; i++)
        if (poly[i] != poly2[i]) { ok12 = 0; break; }
    results[0] = ok12;

    // ── Test 10-bit round-trip ────────────────────────────────────────
    for (int i = 0; i < KYBER_N; i++)
        poly[i] = (int16_t)((i * 13 + 7) % 1024);

    uint8_t buf10[320];
    byte_encode_10(poly, buf10);
    byte_decode_10(buf10, poly2);

    int ok10 = 1;
    for (int i = 0; i < KYBER_N; i++)
        if (poly[i] != poly2[i]) { ok10 = 0; break; }
    results[1] = ok10;

    // ── Test 4-bit round-trip ─────────────────────────────────────────
    for (int i = 0; i < KYBER_N; i++)
        poly[i] = (int16_t)(i % 16);

    uint8_t buf4[128];
    byte_encode_4(poly, buf4);
    byte_decode_4(buf4, poly2);

    int ok4 = 1;
    for (int i = 0; i < KYBER_N; i++)
        if (poly[i] != poly2[i]) { ok4 = 0; break; }
    results[2] = ok4;

    // ── Test 1-bit round-trip ─────────────────────────────────────────
    for (int i = 0; i < KYBER_N; i++)
        poly[i] = (int16_t)(i % 2);

    uint8_t buf1[32];
    byte_encode_1(poly, buf1);
    byte_decode_1(buf1, poly2);

    int ok1 = 1;
    for (int i = 0; i < KYBER_N; i++)
        if (poly[i] != poly2[i]) { ok1 = 0; break; }
    results[3] = ok1;

    // ── Test compress/decompress round-trip ───────────────────────────
    // Compress then decompress loses some precision — check error bound
    // For d=10: max error = Q/(2^11) ~ 1.6, so <= 2
    for (int i = 0; i < KYBER_N; i++)
        poly[i] = (int16_t)((i * 13 + 7) % KYBER_Q);

    int16_t comp[KYBER_N], decomp[KYBER_N];
    compress_poly(poly, comp, 10);
    decompress_poly(comp, decomp, 10);

    int32_t max_err = 0;
    for (int i = 0; i < KYBER_N; i++) {
        int32_t err = (int32_t)poly[i] - (int32_t)decomp[i];
        if (err < 0) err = -err;
        // Handle wrap-around at Q boundary
        if (err > KYBER_Q/2) err = KYBER_Q - err;
        if (err > max_err) max_err = err;
    }
    results[4] = max_err; // should be <= Q/(2^11) ~ 2

    // ── Test compress d=4 error bound ────────────────────────────────
    compress_poly(poly, comp, 4);
    decompress_poly(comp, decomp, 4);

    int32_t max_err4 = 0;
    for (int i = 0; i < KYBER_N; i++) {
        int32_t err = (int32_t)poly[i] - (int32_t)decomp[i];
        if (err < 0) err = -err;
        if (err > KYBER_Q/2) err = KYBER_Q - err;
        if (err > max_err4) max_err4 = err;
    }
    results[5] = max_err4; // should be <= Q/(2^5) ~ 104
}

} // extern "C"
