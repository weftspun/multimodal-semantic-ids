// SPDX-License-Identifier: MIT
// Milestone 3: the TIC EncoderLayer forward in Slang, matched to the Aplus
// torch module (check_enc.py).  Pre-norm transformer block:
//   y  = MHA(norm1(x)) + x
//   out = FFN(norm2(y)) + y
// MHA: per-head softmax(QKᵀ/√dk)·V, h heads; FFN: Linear→ReLU→Linear.
// Small fixed dims prove the architecture; the dims are parameters for training.
static const int S = 4;    // sequence length
static const int D = 8;    // d_model
static const int H = 2;    // heads
static const int DK = 4;   // D / H
static const int F = 16;   // d_ff

[[vk::binding(0, 0)]] RWStructuredBuffer<float> inB;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outB;

// LayerNorm of one D-vector (eps 1e-5, matching torch).
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

[shader("compute")]
[numthreads(1, 1, 1)]
void encoder_test(uint3 tid: SV_DispatchThreadID) {
    int o = 0;
    float X[S * D];
    for (int i = 0; i < S * D; i++) {
        X[i] = inB[o++];
    }
    float g1[D], b1[D];
    for (int i = 0; i < D; i++) {
        g1[i] = inB[o++];
    }
    for (int i = 0; i < D; i++) {
        b1[i] = inB[o++];
    }
    float Wq[D * D], bq[D], Wk[D * D], bk[D], Wv[D * D], bv[D], Wo[D * D], bo[D];
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
    float g2[D], b2[D];
    for (int i = 0; i < D; i++) {
        g2[i] = inB[o++];
    }
    for (int i = 0; i < D; i++) {
        b2[i] = inB[o++];
    }
    float W1[F * D], b1f[F], W2[D * F], b2f[D];
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

    // x2 = norm1(x), per token
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

    // Q, K, V = linear(x2)
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

    // per-head attention → concat AC
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

    // attn output linear, residual with pre-norm input X
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

    // y2 = norm2(y); FFN(y2) + y
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
            hid[j] = max(0.0f, v);  // ReLU
        }
        for (int j = 0; j < D; j++) {
            float v = b2f[j];
            for (int c = 0; c < F; c++) {
                v += hid[c] * W2[j * F + c];
            }
            outB[s * D + j] = v + Y[s * D + j];
        }
    }
}
