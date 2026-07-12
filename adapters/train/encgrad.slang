// SPDX-License-Identifier: MIT
// Milestone 4: reverse-mode autodiff through the EncoderLayer, gradient-checked
// against torch (check_encgrad.py).  The whole pre-norm block is [Differentiable]
// (weights in a LayerW struct); enc_loss = MSE(layer(X), target); bwd_diff yields
// the weight + input gradients, compared to torch.autograd.  Softmax omits the
// max-subtraction: it is shift-invariant, so the gradient equals torch's (whose
// max is detached), and the test inputs are small enough that exp is safe.
static const int S = 4;
static const int D = 8;
static const int H = 2;
static const int DK = 4;
static const int F = 16;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> inB;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outB;

struct LayerW : IDifferentiable {
    float g1[D], b1[D];
    float Wq[D * D], bq[D], Wk[D * D], bk[D], Wv[D * D], bv[D], Wo[D * D], bo[D];
    float g2[D], b2[D];
    float W1[F * D], b1f[F], W2[D * F], b2f[D];
}

[Differentiable]
float[D] layernorm(float x[D], float g[D], float b[D]) {
    float mean = 0.0f;
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        mean += x[i];
    }
    mean /= D;
    float var = 0.0f;
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        var += (x[i] - mean) * (x[i] - mean);
    }
    var /= D;
    float inv = 1.0f / sqrt(var + 1e-5f);
    float y[D];
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        y[i] = (x[i] - mean) * inv * g[i] + b[i];
    }
    return y;
}

[Differentiable]
float[S * D] encoderLayer(float X[S * D], LayerW w) {
    float X2[S * D];
    [ForceUnroll]
    for (int s = 0; s < S; s++) {
        float xi[D];
        [ForceUnroll]
        for (int i = 0; i < D; i++) {
            xi[i] = X[s * D + i];
        }
        float yi[D] = layernorm(xi, w.g1, w.b1);
        [ForceUnroll]
        for (int i = 0; i < D; i++) {
            X2[s * D + i] = yi[i];
        }
    }
    float Q[S * D], K[S * D], V[S * D];
    [ForceUnroll]
    for (int s = 0; s < S; s++) {
        [ForceUnroll]
        for (int j = 0; j < D; j++) {
            float q = w.bq[j], k = w.bk[j], v = w.bv[j];
            [ForceUnroll]
            for (int c = 0; c < D; c++) {
                q += X2[s * D + c] * w.Wq[j * D + c];
                k += X2[s * D + c] * w.Wk[j * D + c];
                v += X2[s * D + c] * w.Wv[j * D + c];
            }
            Q[s * D + j] = q;
            K[s * D + j] = k;
            V[s * D + j] = v;
        }
    }
    float AC[S * D];
    float scale = 1.0f / sqrt((float)DK);
    [ForceUnroll]
    for (int h = 0; h < H; h++) {
        [ForceUnroll]
        for (int si = 0; si < S; si++) {
            float sim[S];
            float ssum = 0.0f;
            [ForceUnroll]
            for (int sj = 0; sj < S; sj++) {
                float dotp = 0.0f;
                [ForceUnroll]
                for (int k = 0; k < DK; k++) {
                    dotp += Q[si * D + h * DK + k] * K[sj * D + h * DK + k];
                }
                sim[sj] = exp(dotp * scale);
                ssum += sim[sj];
            }
            [ForceUnroll]
            for (int k = 0; k < DK; k++) {
                float acc = 0.0f;
                [ForceUnroll]
                for (int sj = 0; sj < S; sj++) {
                    acc += (sim[sj] / ssum) * V[sj * D + h * DK + k];
                }
                AC[si * D + h * DK + k] = acc;
            }
        }
    }
    float Y[S * D];
    [ForceUnroll]
    for (int s = 0; s < S; s++) {
        [ForceUnroll]
        for (int j = 0; j < D; j++) {
            float a = w.bo[j];
            [ForceUnroll]
            for (int c = 0; c < D; c++) {
                a += AC[s * D + c] * w.Wo[j * D + c];
            }
            Y[s * D + j] = a + X[s * D + j];
        }
    }
    float OUT[S * D];
    [ForceUnroll]
    for (int s = 0; s < S; s++) {
        float yi[D];
        [ForceUnroll]
        for (int i = 0; i < D; i++) {
            yi[i] = Y[s * D + i];
        }
        float y2[D] = layernorm(yi, w.g2, w.b2);
        float hid[F];
        [ForceUnroll]
        for (int j = 0; j < F; j++) {
            float v = w.b1f[j];
            [ForceUnroll]
            for (int c = 0; c < D; c++) {
                v += y2[c] * w.W1[j * D + c];
            }
            hid[j] = max(0.0f, v);
        }
        [ForceUnroll]
        for (int j = 0; j < D; j++) {
            float v = w.b2f[j];
            [ForceUnroll]
            for (int c = 0; c < F; c++) {
                v += hid[c] * w.W2[j * F + c];
            }
            OUT[s * D + j] = v + Y[s * D + j];
        }
    }
    return OUT;
}

[Differentiable]
float enc_loss(float X[S * D], LayerW w, no_diff float target[S * D]) {
    float Y[S * D] = encoderLayer(X, w);
    float s = 0.0f;
    [ForceUnroll]
    for (int i = 0; i < S * D; i++) {
        s += (Y[i] - target[i]) * (Y[i] - target[i]);
    }
    return s / (S * D);
}

LayerW loadLayer(inout int o) {
    LayerW w;
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        w.g1[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        w.b1[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D * D; i++) {
        w.Wq[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        w.bq[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D * D; i++) {
        w.Wk[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        w.bk[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D * D; i++) {
        w.Wv[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        w.bv[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D * D; i++) {
        w.Wo[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        w.bo[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        w.g2[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        w.b2[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < F * D; i++) {
        w.W1[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < F; i++) {
        w.b1f[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D * F; i++) {
        w.W2[i] = inB[o++];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        w.b2f[i] = inB[o++];
    }
    return w;
}

void storeLayerGrad(LayerW.Differential d, inout int o) {
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        outB[o++] = d.g1[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        outB[o++] = d.b1[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D * D; i++) {
        outB[o++] = d.Wq[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        outB[o++] = d.bq[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D * D; i++) {
        outB[o++] = d.Wk[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        outB[o++] = d.bk[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D * D; i++) {
        outB[o++] = d.Wv[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        outB[o++] = d.bv[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D * D; i++) {
        outB[o++] = d.Wo[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        outB[o++] = d.bo[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        outB[o++] = d.g2[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        outB[o++] = d.b2[i];
    }
    [ForceUnroll]
    for (int i = 0; i < F * D; i++) {
        outB[o++] = d.W1[i];
    }
    [ForceUnroll]
    for (int i = 0; i < F; i++) {
        outB[o++] = d.b1f[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D * F; i++) {
        outB[o++] = d.W2[i];
    }
    [ForceUnroll]
    for (int i = 0; i < D; i++) {
        outB[o++] = d.b2f[i];
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void enc_grad(uint3 tid: SV_DispatchThreadID) {
    int o = 0;
    float X[S * D];
    [ForceUnroll]
    for (int i = 0; i < S * D; i++) {
        X[i] = inB[o++];
    }
    LayerW w = loadLayer(o);
    float target[S * D];
    [ForceUnroll]
    for (int i = 0; i < S * D; i++) {
        target[i] = inB[o++];
    }

    var dpX = diffPair(X);
    var dpw = diffPair(w);
    bwd_diff(enc_loss)(dpX, dpw, target, 1.0f);

    int wo = 0;
    outB[wo++] = enc_loss(X, w, target);  // forward loss cross-check
    storeLayerGrad(dpw.d, wo);
    [ForceUnroll]
    for (int i = 0; i < S * D; i++) {
        outB[wo++] = dpX.d[i];
    }
}
