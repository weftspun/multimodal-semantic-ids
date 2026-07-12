"""TIC assembled from our slangtorch ops (tk_ops) — mirrors my_model.TIC and the Vulkan netfwd graph.

Pre-norm encoder layers, Embedder scaled by √d_model, two TPM heads (one encoder layer → sequence
mean → linear).  Weights are plain CUDA tensors (optionally requires_grad) keyed by the same names as
the torch state_dict, so we can load a reference TIC's weights and check fwd/bwd parity.  Operates on
one window x:(S, NIN); batch by looping or (T7) stacking rows.
"""
import math
import torch
import tk_ops as tk


class TICKernels:
    def __init__(self, state_dict, stack, D, H, F, requires_grad=False):
        self.stack, self.D, self.H, self.F = stack, D, H, F
        self.w = {}
        for k, v in state_dict.items():
            t = v.detach().to("cuda", torch.float32).clone()
            if requires_grad:
                t.requires_grad_(True)
            self.w[k] = t

    def params(self):
        return self.w

    def _lin(self, x, p):  # nn.Linear: weight (out,in), bias (out)  ->  x·Wᵀ + b
        return tk.Linear.apply(x, self.w[p + ".weight"], self.w[p + ".bias"])

    def _encoder(self, x, p):  # pre-norm EncoderLayer (matches check_tic ENC order)
        x1 = tk.LayerNorm.apply(x, self.w[p + ".norm1.weight"], self.w[p + ".norm1.bias"])
        Q = self._lin(x1, p + ".mha.q_linear")
        K = self._lin(x1, p + ".mha.k_linear")
        V = self._lin(x1, p + ".mha.v_linear")
        a = tk.Attention.apply(Q, K, V, self.H)
        x = x + self._lin(a, p + ".mha.output")          # residual
        x2 = tk.LayerNorm.apply(x, self.w[p + ".norm2.weight"], self.w[p + ".norm2.bias"])
        ff = self._lin(tk.ReLU.apply(self._lin(x2, p + ".mlp.fc1")), p + ".mlp.fc2")
        return x + ff                                    # residual

    def _head(self, x, p):  # TPM: encoder layer -> seq mean -> linear
        h = self._encoder(x, p + ".encoder")
        h = h.mean(dim=0, keepdim=True)                  # (1, D)
        return self._lin(h, p + ".mapping").squeeze(0)   # (NOUT,)

    def forward(self, x):  # x:(S, NIN) -> (global_shift, local_shift) each (NOUT,)
        x = self._lin(x, "input_embedding_layer.embed") * math.sqrt(self.D)
        for i in range(self.stack):
            x = self._encoder(x, f"encoder_banckbone.{i}")
        return self._head(x, "TPM_global"), self._head(x, "TPM_local")

    # ── batched path (T7): X:(B,S,NIN); row-wise ops run on (B·S) rows, attention stays per-window ─
    def _encoder_b(self, x, p, B, S):  # x:(B·S, D)
        x1 = tk.LayerNorm.apply(x, self.w[p + ".norm1.weight"], self.w[p + ".norm1.bias"])
        Q = self._lin(x1, p + ".mha.q_linear")
        K = self._lin(x1, p + ".mha.k_linear")
        V = self._lin(x1, p + ".mha.v_linear")
        a = tk.AttentionB.apply(Q, K, V, self.H, B, S)
        x = x + self._lin(a, p + ".mha.output")
        x2 = tk.LayerNorm.apply(x, self.w[p + ".norm2.weight"], self.w[p + ".norm2.bias"])
        ff = self._lin(tk.ReLU.apply(self._lin(x2, p + ".mlp.fc1")), p + ".mlp.fc2")
        return x + ff

    def _head_b(self, x, p, B, S):
        h = self._encoder_b(x, p + ".encoder", B, S)
        h = h.view(B, S, self.D).mean(dim=1)             # (B, D)
        return self._lin(h, p + ".mapping")              # (B, NOUT)

    def forward_batch(self, X):  # X:(B,S,NIN) -> (global, local) each (B, NOUT)
        B, S, NIN = X.shape
        x = self._lin(X.reshape(B * S, NIN), "input_embedding_layer.embed") * math.sqrt(self.D)
        for i in range(self.stack):
            x = self._encoder_b(x, f"encoder_banckbone.{i}", B, S)
        return self._head_b(x, "TPM_global", B, S), self._head_b(x, "TPM_local", B, S)
