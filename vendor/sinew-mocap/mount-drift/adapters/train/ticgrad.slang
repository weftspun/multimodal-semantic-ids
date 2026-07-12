// SPDX-License-Identifier: MIT
// Milestone 4b: reverse-mode autodiff through the whole TIC network, gradient-
// checked against torch (check_ticgrad.py).  The full forward is [Differentiable]
// over a Weights struct (embed + STACK encoder layers + two TPM heads); the loss
// is MSE(cat(global, local), target); bwd_diff yields every weight gradient.
static const int S = 4;
static const int NIN = 12;
static const int D = 8;
static const int H = 2;
static const int DK = 4;
static const int F = 16;
static const int NOUT = 6;
static const int STACK = 3;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> inB;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> outB;

struct LayerW : IDifferentiable {
    float g1[D], b1[D];
    float Wq[D * D], bq[D], Wk[D * D], bk[D], Wv[D * D], bv[D], Wo[D * D], bo[D];
    float g2[D], b2[D];
    float W1[F * D], b1f[F], W2[D * F], b2f[D];
}

struct Weights : IDifferentiable {
    float We[D * NIN], be[D];
    LayerW layers[STACK];
    LayerW gHead;
    float gMap[NOUT * D], gMapB[NOUT];
    LayerW lHead;
    float lMap[NOUT * D], lMapB[NOUT];
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
float[NOUT] tpmHead(float X[S * D], LayerW w, float Wm[NOUT * D], float bm[NOUT]) {
    float Y[S * D] = encoderLayer(X, w);
    float meanv[D];
    [ForceUnroll]
    for (int j = 0; j < D; j++) {
        float s = 0.0f;
        [ForceUnroll]
        for (int t = 0; t < S; t++) {
            s += Y[t * D + j];
        }
        meanv[j] = s / S;
    }
    float out[NOUT];
    [ForceUnroll]
    for (int j = 0; j < NOUT; j++) {
        float v = bm[j];
        [ForceUnroll]
        for (int c = 0; c < D; c++) {
            v += meanv[c] * Wm[j * D + c];
        }
        out[j] = v;
    }
    return out;
}

[Differentiable]
float tic_loss(Weights w, no_diff float x[S * NIN], no_diff float target[2 * NOUT]) {
    float scale = sqrt((float)D);
    float X[S * D];
    [ForceUnroll]
    for (int s = 0; s < S; s++) {
        [ForceUnroll]
        for (int j = 0; j < D; j++) {
            float v = w.be[j];
            [ForceUnroll]
            for (int c = 0; c < NIN; c++) {
                v += x[s * NIN + c] * w.We[j * NIN + c];
            }
            X[s * D + j] = v * scale;
        }
    }
    [ForceUnroll]
    for (int l = 0; l < STACK; l++) {
        X = encoderLayer(X, w.layers[l]);
    }
    float og[NOUT] = tpmHead(X, w.gHead, w.gMap, w.gMapB);
    float ol[NOUT] = tpmHead(X, w.lHead, w.lMap, w.lMapB);
    float s = 0.0f;
    [ForceUnroll]
    for (int j = 0; j < NOUT; j++) {
        s += (og[j] - target[j]) * (og[j] - target[j]);
        s += (ol[j] - target[NOUT + j]) * (ol[j] - target[NOUT + j]);
    }
    return s / (2 * NOUT);
}

LayerW loadLayer(inout int o) {
    LayerW w;
    [ForceUnroll] for (int i = 0; i < D; i++) { w.g1[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { w.b1[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D * D; i++) { w.Wq[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { w.bq[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D * D; i++) { w.Wk[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { w.bk[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D * D; i++) { w.Wv[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { w.bv[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D * D; i++) { w.Wo[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { w.bo[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { w.g2[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { w.b2[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < F * D; i++) { w.W1[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < F; i++) { w.b1f[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D * F; i++) { w.W2[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { w.b2f[i] = inB[o++]; }
    return w;
}

void storeLayerGrad(LayerW.Differential d, inout int o) {
    [ForceUnroll] for (int i = 0; i < D; i++) { outB[o++] = d.g1[i]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { outB[o++] = d.b1[i]; }
    [ForceUnroll] for (int i = 0; i < D * D; i++) { outB[o++] = d.Wq[i]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { outB[o++] = d.bq[i]; }
    [ForceUnroll] for (int i = 0; i < D * D; i++) { outB[o++] = d.Wk[i]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { outB[o++] = d.bk[i]; }
    [ForceUnroll] for (int i = 0; i < D * D; i++) { outB[o++] = d.Wv[i]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { outB[o++] = d.bv[i]; }
    [ForceUnroll] for (int i = 0; i < D * D; i++) { outB[o++] = d.Wo[i]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { outB[o++] = d.bo[i]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { outB[o++] = d.g2[i]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { outB[o++] = d.b2[i]; }
    [ForceUnroll] for (int i = 0; i < F * D; i++) { outB[o++] = d.W1[i]; }
    [ForceUnroll] for (int i = 0; i < F; i++) { outB[o++] = d.b1f[i]; }
    [ForceUnroll] for (int i = 0; i < D * F; i++) { outB[o++] = d.W2[i]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { outB[o++] = d.b2f[i]; }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void tic_grad(uint3 tid: SV_DispatchThreadID) {
    int o = 0;
    float x[S * NIN];
    [ForceUnroll] for (int i = 0; i < S * NIN; i++) { x[i] = inB[o++]; }
    Weights w;
    [ForceUnroll] for (int i = 0; i < D * NIN; i++) { w.We[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { w.be[i] = inB[o++]; }
    [ForceUnroll] for (int l = 0; l < STACK; l++) { w.layers[l] = loadLayer(o); }
    w.gHead = loadLayer(o);
    [ForceUnroll] for (int i = 0; i < NOUT * D; i++) { w.gMap[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < NOUT; i++) { w.gMapB[i] = inB[o++]; }
    w.lHead = loadLayer(o);
    [ForceUnroll] for (int i = 0; i < NOUT * D; i++) { w.lMap[i] = inB[o++]; }
    [ForceUnroll] for (int i = 0; i < NOUT; i++) { w.lMapB[i] = inB[o++]; }
    float target[2 * NOUT];
    [ForceUnroll] for (int i = 0; i < 2 * NOUT; i++) { target[i] = inB[o++]; }

    var dpw = diffPair(w);
    bwd_diff(tic_loss)(dpw, x, target, 1.0f);

    int wo = 0;
    outB[wo++] = tic_loss(w, x, target);
    [ForceUnroll] for (int i = 0; i < D * NIN; i++) { outB[wo++] = dpw.d.We[i]; }
    [ForceUnroll] for (int i = 0; i < D; i++) { outB[wo++] = dpw.d.be[i]; }
    [ForceUnroll] for (int l = 0; l < STACK; l++) { storeLayerGrad(dpw.d.layers[l], wo); }
    storeLayerGrad(dpw.d.gHead, wo);
    [ForceUnroll] for (int i = 0; i < NOUT * D; i++) { outB[wo++] = dpw.d.gMap[i]; }
    [ForceUnroll] for (int i = 0; i < NOUT; i++) { outB[wo++] = dpw.d.gMapB[i]; }
    storeLayerGrad(dpw.d.lHead, wo);
    [ForceUnroll] for (int i = 0; i < NOUT * D; i++) { outB[wo++] = dpw.d.lMap[i]; }
    [ForceUnroll] for (int i = 0; i < NOUT; i++) { outB[wo++] = dpw.d.lMapB[i]; }
}
