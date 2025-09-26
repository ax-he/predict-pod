#include <openssl/sha.h>
#include <openssl/cmac.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void do_sha256(const unsigned char* buf, size_t len, unsigned char out[32]) {
    SHA256_CTX c; SHA256_Init(&c);
    size_t off = 0;
    while (off < len) {
        size_t l = (len - off > 8192) ? 8192 : (len - off);
        SHA256_Update(&c, buf + off, l);
        off += l;
    }
    SHA256_Final(out, &c);
}

static void do_cmac(const unsigned char* key, const unsigned char* buf, size_t len, unsigned char mac[16]) {
    CMAC_CTX* ctx = CMAC_CTX_new();
    CMAC_Init(ctx, key, 16, EVP_aes_128_cbc(), NULL);
    size_t off = 0; size_t maclen = 0;
    while (off < len) {
        size_t l = (len - off > 4096) ? 4096 : (len - off);
        CMAC_Update(ctx, buf + off, l);
        off += l;
    }
    CMAC_Final(ctx, mac, &maclen);
    CMAC_CTX_free(ctx);
}

int main() {
    const size_t N = 16 * 1024 * 1024; // 16 MiB payload
    unsigned char *buf = (unsigned char*) aligned_alloc(64, N);
    if (!buf) { perror("alloc"); return 1; }
    RAND_bytes(buf, N);

    // mixed scalar work to avoid optimizing-away
    volatile float acc = 0.0f;
    for (size_t i = 0; i < (1<<20); i++) acc += expf((float)(i & 1023) * 1e-3f);

    // sha256 + cmac
    unsigned char out[32], mac[16], key[16];
    RAND_bytes((unsigned char*)key, sizeof(key));
    do_sha256(buf, N, out);
    do_cmac(key, buf, N, mac);

    // some memcpys
    unsigned char *tmp = (unsigned char*) aligned_alloc(64, N);
    memcpy(tmp, buf, N);
    memcpy(buf, tmp, N/2);

    printf("done acc=%f out0=%02x mac0=%02x\n", acc, out[0], mac[0]);
    free(tmp); free(buf);
    return 0;
}
