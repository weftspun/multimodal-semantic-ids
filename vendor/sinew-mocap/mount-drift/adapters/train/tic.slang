// SPDX-License-Identifier: MIT
// Milestone 3b: the full TIC forward in Slang, matched to torch's TIC.forward
// (check_tic.py).  Embedder (Linear·√D) → STACK backbone encoder layers → two
// TPM heads (encoder layer → mean over sequence → mapping linear), emitting the
// global and local 6D-per-sensor shift.  Small dims prove the assembly; the dims
// are parameters for training.
static const int S = 4;      // sequence length
static const int NIN = 12;   // n_input
static const int D = 8;      // d_model
static const int H = 2;      // heads
static const int DK = 4;     // D / H
static const int F = 16;     // d_ff
static const int NOUT = 6;   // n_output
static const int STACK = 3;  // backbone layers
static const int WL = 4 * D * D + 2 * F * D + 9 * D + F;  // weights per encoder layer

[[vk::binding(0, 0)]] RWStructuredBuffer<float> inB;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outB;

void layernorm(float x[D], float g[D], float b[D], out float y[D]) {
    float mean = 0.0f;
    for (int i = 0; i < D; i++) {
        mean += x[i];
    }
    mean /= D;
    float var = 0.0f;
    for (int i = 0; i < D; i++) {
        var += (x[i] - mean) * (x[i] - mean);
    }
    var /= D;
    float inv = 1.0f / sqrt(var + 1e-5f);
    for (int i = 0; i < D; i++) {
        y[i] = (x[i] - mean) * inv * g[i] + b[i];
    }
}

// One pre-norm EncoderLayer over X (S tokens of D), weights read from inB at `base`.
void encoderLayer(inout float X[S * D], int base) {
    int o = base;
    float g1[D], b1[D], Wq[D * D], bq[D], Wk[D * D], bk[D], Wv[D * D], bv[D], Wo[D * D], bo[D];
    float g2[D], b2[D], W1[F * D], b1f[F], W2[D * F], b2f[D];
    for (int i = 0; i < D; i++) {
        g1[i] = inB[o++];
    }
    for (int i = 0; i < D; i++) {
        b1[i] = inB[o++];
    }
    for (int i = 0; i < D * D; i++) {
        Wq[i] = inB[o++];
    }
    for (int i = 0; i < D; i++) {
        bq[i] = inB[o++];
    }
    for (int i = 0; i < D * D; i++) {
        Wk[i] = inB[o++];
    }
    for (int i = 0; i < D; i++) {
        bk[i] = inB[o++];
    }
    for (int i = 0; i < D * D; i++) {
        Wv[i] = inB[o++];
    }
    for (int i = 0; i < D; i++) {
        bv[i] = inB[o++];
    }
    for (int i = 0; i < D * D; i++) {
        Wo[i] = inB[o++];
    }
    for (int i = 0; i < D; i++) {
        bo[i] = inB[o++];
    }
    for (int i = 0; i < D; i++) {
        g2[i] = inB[o++];
    }
    for (int i = 0; i < D; i++) {
        b2[i] = inB[o++];
    }
    for (int i = 0; i < F * D; i++) {
        W1[i] = inB[o++];
    }
    for (int i = 0; i < F; i++) {
        b1f[i] = inB[o++];
    }
    for (int i = 0; i < D * F; i++) {
        W2[i] = inB[o++];
    }
    for (int i = 0; i < D; i++) {
        b2f[i] = inB[o++];
    }

    float X2[S * D];
    for (int s = 0; s < S; s++) {
        float xi[D], yi[D];
        for (int i = 0; i < D; i++) {
            xi[i] = X[s * D + i];
        }
        layernorm(xi, g1, b1, yi);
        for (int i = 0; i < D; i++) {
            X2[s * D + i] = yi[i];
        }
    }
    float Q[S * D], K[S * D], V[S * D];
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < D; j++) {
            float q = bq[j], k = bk[j], v = bv[j];
            for (int c = 0; c < D; c++) {
                q += X2[s * D + c] * Wq[j * D + c];
                k += X2[s * D + c] * Wk[j * D + c];
                v += X2[s * D + c] * Wv[j * D + c];
            }
            Q[s * D + j] = q;
            K[s * D + j] = k;
            V[s * D + j] = v;
        }
    }
    float AC[S * D];
    float scale = 1.0f / sqrt((float)DK);
    for (int h = 0; h < H; h++) {
        for (int si = 0; si < S; si++) {
            float sim[S];
            float m = -1e30f;
            for (int sj = 0; sj < S; sj++) {
                float dotp = 0.0f;
                for (int k = 0; k < DK; k++) {
                    dotp += Q[si * D + h * DK + k] * K[sj * D + h * DK + k];
                }
                sim[sj] = dotp * scale;
                m = max(m, sim[sj]);
            }
            float ssum = 0.0f;
            for (int sj = 0; sj < S; sj++) {
                sim[sj] = exp(sim[sj] - m);
                ssum += sim[sj];
            }
            for (int k = 0; k < DK; k++) {
                float acc = 0.0f;
                for (int sj = 0; sj < S; sj++) {
                    acc += (sim[sj] / ssum) * V[sj * D + h * DK + k];
                }
                AC[si * D + h * DK + k] = acc;
            }
        }
    }
    float Y[S * D];
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < D; j++) {
            float a = bo[j];
            for (int c = 0; c < D; c++) {
                a += AC[s * D + c] * Wo[j * D + c];
            }
            Y[s * D + j] = a + X[s * D + j];
        }
    }
    for (int s = 0; s < S; s++) {
        float yi[D], y2[D];
        for (int i = 0; i < D; i++) {
            yi[i] = Y[s * D + i];
        }
        layernorm(yi, g2, b2, y2);
        float hid[F];
        for (int j = 0; j < F; j++) {
            float v = b1f[j];
            for (int c = 0; c < D; c++) {
                v += y2[c] * W1[j * D + c];
            }
            hid[j] = max(0.0f, v);
        }
        for (int j = 0; j < D; j++) {
            float v = b2f[j];
            for (int c = 0; c < F; c++) {
                v += hid[c] * W2[j * F + c];
            }
            X[s * D + j] = v + Y[s * D + j];
        }
    }
}

// One TPM head: a copy of X through one encoder layer, mean over the sequence,
// then a mapping linear (NOUT×D) → outB[outOff..].
void tpmHead(float Xin[S * D], int layerBase, int mapBase, int outOff) {
    float X[S * D];
    for (int i = 0; i < S * D; i++) {
        X[i] = Xin[i];
    }
    encoderLayer(X, layerBase);
    float meanv[D];
    for (int j = 0; j < D; j++) {
        float s = 0.0f;
        for (int t = 0; t < S; t++) {
            s += X[t * D + j];
        }
        meanv[j] = s / S;
    }
    int o = mapBase;
    float Wm[NOUT * D];
    for (int i = 0; i < NOUT * D; i++) {
        Wm[i] = inB[o++];
    }
    float bm[NOUT];
    for (int i = 0; i < NOUT; i++) {
        bm[i] = inB[o++];
    }
    for (int j = 0; j < NOUT; j++) {
        float v = bm[j];
        for (int c = 0; c < D; c++) {
            v += meanv[c] * Wm[j * D + c];
        }
        outB[outOff + j] = v;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void tic_test(uint3 tid: SV_DispatchThreadID) {
    // layout: x(S*NIN) We(D*NIN) be(D) [STACK × WL] gHead(WL) gMap(NOUT*D+NOUT)
    //         lHead(WL) lMap(NOUT*D+NOUT)
    int o = 0;
    float xin[S * NIN];
    for (int i = 0; i < S * NIN; i++) {
        xin[i] = inB[o++];
    }
    float We[D * NIN];
    for (int i = 0; i < D * NIN; i++) {
        We[i] = inB[o++];
    }
    float be[D];
    for (int i = 0; i < D; i++) {
        be[i] = inB[o++];
    }
    float scale = sqrt((float)D);
    float X[S * D];
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < D; j++) {
            float v = be[j];
            for (int c = 0; c < NIN; c++) {
                v += xin[s * NIN + c] * We[j * NIN + c];
            }
            X[s * D + j] = v * scale;
        }
    }
    int base = o;  // first backbone layer weights begin here
    for (int l = 0; l < STACK; l++) {
        encoderLayer(X, base + l * WL);
    }
    int gBase = base + STACK * WL;
    int gMap = gBase + WL;
    int lBase = gMap + NOUT * D + NOUT;
    int lMap = lBase + WL;
    tpmHead(X, gBase, gMap, 0);        // global_shift → outB[0..NOUT)
    tpmHead(X, lBase, lMap, NOUT);     // local_shift  → outB[NOUT..2*NOUT)
}
