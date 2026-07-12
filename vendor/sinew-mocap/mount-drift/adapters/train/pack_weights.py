# SPDX-License-Identifier: MIT
# The single source of truth for the TIC weight layout that caltrain.exe / netfwd.exe read.
# A flat little-endian float32 blob in the order the engine-free C++ load() walks the net:
#   embed -> STACK × encoder_banckbone.{i} -> TPM_global.encoder + mapping -> TPM_local.encoder + mapping
# Each encoder layer is the 16 tensors of ENC, in this exact order.  Used to write winit.bin (initial
# weights) and to re-pack a trained torch state_dict into wtrained.bin form.  The "encoder_banckbone"
# spelling is the model's state_dict key, not a typo here — do not "fix" it.
import numpy as np

# One pre-norm EncoderLayer, in caltrain's read order.
ENC = ["norm1.weight", "norm1.bias",
       "mha.q_linear.weight", "mha.q_linear.bias", "mha.k_linear.weight", "mha.k_linear.bias",
       "mha.v_linear.weight", "mha.v_linear.bias", "mha.output.weight", "mha.output.bias",
       "norm2.weight", "norm2.bias",
       "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"]


def pack_winit(state_dict, stack):
    """Flatten a TIC state_dict into the caltrain/netfwd weight blob (np.float32, 1-D)."""
    def W(n):
        v = state_dict[n]
        a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        return a.ravel()

    def enc(p):
        return np.concatenate([W(f"{p}.{n}") for n in ENC])

    parts = [W("input_embedding_layer.embed.weight"), W("input_embedding_layer.embed.bias")]
    for i in range(stack):
        parts.append(enc(f"encoder_banckbone.{i}"))
    parts.append(enc("TPM_global.encoder"))
    parts += [W("TPM_global.mapping.weight"), W("TPM_global.mapping.bias")]
    parts.append(enc("TPM_local.encoder"))
    parts += [W("TPM_local.mapping.weight"), W("TPM_local.mapping.bias")]
    return np.concatenate(parts).astype(np.float32)
