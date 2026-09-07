#!/usr/bin/env python3
"""Compare Vontra 4-bit MLX checkpoint (dequantized) against the official HF BF16
tensors fetched by HTTP range requests (only layer 0 + globals, expert 0 only).

Revue 2026-09-02. Necessite `hf auth token` valide et mlx (venv 0.6.17 ou python systeme).
Les tenseurs BF16 (~120 Mo) sont mis en cache dans Scripts/hf-cache/ (ou $QWEN4_HF_CACHE).
Resultat attendu : projections 4-bit rel_rms ~0.085 / cos ~0.996 ; hc_norm : ecart +1.0."""
import json, os, struct, subprocess, sys
import numpy as np
import mlx.core as mx

S = os.environ.get("QWEN4_HF_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf-cache"))
os.makedirs(S, exist_ok=True)
M = "/Volumes/Lexar/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"
BASE = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
TOK = subprocess.run(["hf", "auth", "token"], capture_output=True, text=True).stdout.strip()
idx = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]


def header(shard):
    out = f"{S}/hdr_{shard}.bin"
    if not os.path.exists(out):
        subprocess.run(["curl", "-sL", "-H", f"Authorization: Bearer {TOK}", "-r", "0-400000",
                        BASE + shard + ".safetensors", "-o", out], check=True)
    b = open(out, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    return json.loads(b[8:8 + n]), 8 + n


def fetch(shard, name, nbytes=None):
    h, base = header(shard)
    v = h[name]
    a, b = v["data_offsets"]
    if nbytes is not None:
        b = a + nbytes
    out = f"{S}/hf_{name}.bin"
    if not os.path.exists(out) or os.path.getsize(out) != (b - a):
        subprocess.run(["curl", "-sL", "-H", f"Authorization: Bearer {TOK}", "-r", f"{base + a}-{base + b - 1}",
                        BASE + shard + ".safetensors", "-o", out], check=True)
    raw = np.frombuffer(open(out, "rb").read(), dtype=np.uint16)
    return mx.array(raw.astype(np.uint32) << 16).view(mx.float32)  # bf16 -> f32 exact


def mlx_tensor(key):
    return mx.load(f"{M}/{idx[key]}")[key]


def deq(key):
    w = mlx_tensor(key + ".weight")
    s = mlx_tensor(key + ".scales")
    b = mlx_tensor(key + ".biases")
    return mx.dequantize(w, s, b, group_size=32, bits=4).astype(mx.float32)


def report(label, hf, mlxv):
    hf = hf.reshape(-1); mlxv = mlxv.reshape(-1)
    if hf.shape != mlxv.shape:
        print(f"{label:60s} SHAPE MISMATCH hf {hf.shape} mlx {mlxv.shape}"); return
    d = hf - mlxv
    rel = float(mx.sqrt(mx.mean(d * d)) / (mx.sqrt(mx.mean(hf * hf)) + 1e-12))
    cos = float((hf @ mlxv) / (mx.linalg.norm(hf) * mx.linalg.norm(mlxv) + 1e-12))
    print(f"{label:60s} rel_rms={rel:.4f} cos={cos:.5f} max|d|={float(mx.abs(d).max()):.4g} hf_rms={float(mx.sqrt(mx.mean(hf*hf))):.4g}")


P = "model.language_model.layers.0."
Q = "language_model.model.layers.0."
# --- exact (bf16) tensors: hyper-connections, norms, A_log, dt_bias, conv1d, router gate
for name, key, tf in [
    ("attn_hyper_connection.block_inject_weight.weight", "attn_hyper_connection.block_inject_weight.weight", None),
    ("attn_hyper_connection.hc_norm.weight", "attn_hyper_connection.hc_norm.weight", None),
    ("linear_attn.A_log", "linear_attn.A_log", None),
    ("linear_attn.dt_bias", "linear_attn.dt_bias", None),
    ("linear_attn.norm.weight", "linear_attn.norm.weight", None),
    ("linear_attn.conv1d.weight", "linear_attn.conv1d.weight", "conv"),
]:
    hf = fetch("model-00001-of-00131", P + name)
    mv = mlx_tensor(Q + key).astype(mx.float32)
    if tf == "conv":
        hf = hf.reshape(10240, 1, 4)
        report("conv1d as-is [10240,4,1] vs hf[10240,1,4]", hf, mv)
        report("conv1d hf.moveaxis(2,1)", mx.moveaxis(hf, 2, 1), mv)
    else:
        report(name, hf, mv)
hf = fetch("model-00003-of-00131", P + "mlp.gate.weight")
report("mlp.gate.weight (router, bf16)", hf, mlx_tensor(Q + "mlp.gate.weight").astype(mx.float32))

# --- quantized tensors
for name in ["attn_hyper_connection.input_mix_weight_down.weight", "attn_hyper_connection.input_mix_weight_up.weight",
             "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight", "linear_attn.in_proj_qkv.weight",
             "linear_attn.in_proj_z.weight", "linear_attn.out_proj.weight"]:
    hf = fetch("model-00001-of-00131", P + name)
    key = Q + name[: -len(".weight")]
    if (key + ".scales") in idx:
        report(name + " (deq 4-bit)", hf, deq(key))
    else:
        report(name + " (bf16)", hf, mlx_tensor(key + ".weight").astype(mx.float32))
for name in ["mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight", "mlp.shared_expert_gate.weight"]:
    hf = fetch("model-00003-of-00131", P + name)
    key = Q + name[: -len(".weight")]
    report(name + (" (deq)" if key + ".scales" in idx else " (bf16)"), hf, deq(key) if key + ".scales" in idx else mlx_tensor(key + ".weight").astype(mx.float32))

# --- expert 0: HF gate_up_proj[0] = [1280, 2560]; down_proj[0] = [2560, 640]
gu = fetch("model-00002-of-00131", P + "mlp.experts.gate_up_proj", nbytes=1280 * 2560 * 2).reshape(1280, 2560)
dn = fetch("model-00003-of-00131", P + "mlp.experts.down_proj", nbytes=2560 * 640 * 2).reshape(2560, 640)
def deq_expert(key, e):
    w = mlx_tensor(key + ".weight")[e]; s = mlx_tensor(key + ".scales")[e]; b = mlx_tensor(key + ".biases")[e]
    return mx.dequantize(w, s, b, group_size=32, bits=4).astype(mx.float32)
g0 = deq_expert(Q + "mlp.switch_mlp.gate_proj", 0)
u0 = deq_expert(Q + "mlp.switch_mlp.up_proj", 0)
d0 = deq_expert(Q + "mlp.switch_mlp.down_proj", 0)
print("mlx expert0 shapes", g0.shape, u0.shape, d0.shape)
report("expert0 gate = hf gate_up[:640]", gu[:640], g0)
report("expert0 up   = hf gate_up[640:]", gu[640:], u0)
report("expert0 gate vs hf gate_up[640:] (swap test)", gu[640:], g0)
report("expert0 down = hf down_proj[0]", dn, d0)
report("expert0 down vs hf down_proj[0].T", dn.T, d0)
