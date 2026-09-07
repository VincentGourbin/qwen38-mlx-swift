#!/usr/bin/env python3
"""Teacher-forced Flash-Next logits using the OFFICIAL mlx-vlm 0.6.17 qwen4_exp
package (no vendored reference, no compat shims), layer by layer.

Requires a venv with `pip install mlx-vlm==0.6.17` (revue 2026-09-02 :
/private/tmp/claude-501/.../scratchpad/venv617 ou en recréer un).

Usage: python Scripts/qwen4-exp-official-teacher-forced.py --model-dir DIR \
    --ids 1,2,3 --output out.safetensors [--norm-shift -1] [--seed N] [--layers 0,1,2]

--norm-shift -1 : le checkpoint Vontra stocke les normes zero-centrees deja
+1 (convention sanitize qwen3_5) ; le runtime applique encore (1 + w).
Avec -1, les logits deviennent coherents (voir PLAN.md, revue 2026-09-02).
Le script imprime le hit-rate teacher-forced sur le prompt lui-meme : c'est le
test de sanite absolu qui manquait a toutes les parites precedentes.
"""
import argparse, gc, json, time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_vlm
import mlx_vlm.models.qwen4_exp.language as L
import mlx_vlm.models.qwen4_exp.config as C


def local_key(key: str) -> str:
    marker = ".ngram_embedding.shard_"
    if marker in key:
        head, suffix = key.split(marker, 1)
        return f"{head}.ngram_embedding.shards.{suffix}"
    return key


def quantize_by_keys(module, local_keys, q):
    qpaths = {k[: -len(".weight")] for k in local_keys
              if k.endswith(".weight") and k[: -len(".weight")] + ".scales" in local_keys}
    nn.quantize(module, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"),
                class_predicate=lambda p, m: p in qpaths)
    return qpaths


def load_prefix(model_dir, index, prefix, strip):
    selected = {k: s for k, s in index.items() if k.startswith(prefix)}
    weights = {}
    for shard in sorted(set(selected.values())):
        arrays = mx.load(str(model_dir / shard))
        for k, s in selected.items():
            if s == shard:
                weights[local_key(k[len(strip):])] = arrays[k]
    return weights


class GlobalWeights(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.hyper_connection_mixer = L.Qwen4ExpGatedResidual(config, use_combine=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None, help="override text_config.seed (default: package default)")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--swap-gate-up", action="store_true")
    ap.add_argument("--norm-shift", type=float, default=0.0, help="add this to zero-centered norm weights at load (e.g. -1)")
    args = ap.parse_args()

    raw = json.loads((args.model_dir / "config.json").read_text())
    q = raw.get("quantization") or {"group_size": 32, "bits": 4, "mode": "affine"}
    text_raw = dict(raw["text_config"])
    if args.seed is not None:
        text_raw["seed"] = args.seed
    config = C.TextConfig.from_dict(text_raw)
    print(f"mlx {mx.__version__} mlx_vlm {mlx_vlm.__version__} seed={config.seed} "
          f"ple_layer_ids={config.ple_layer_ids} norm_topk_prob={config.norm_topk_prob} "
          f"hc_lowrank={config.hc_lowrank} q={q}", flush=True)
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    ids = [int(x) for x in args.ids.split(",") if x.strip()]
    input_ids = mx.array([ids], dtype=mx.int32)

    # globals
    g = GlobalWeights(config)
    gw = {}
    for k, s in index.items():
        if k.startswith("language_model.model.embed_tokens.") or k.startswith("language_model.model.hyper_connection_mixer.") or k.startswith("language_model.lm_head."):
            gw[k] = s
    weights = {}
    for shard in sorted(set(gw.values())):
        arrays = mx.load(str(args.model_dir / shard))
        for k, s in gw.items():
            if s != shard:
                continue
            local = k[len("language_model.model."):] if k.startswith("language_model.model.") else k[len("language_model."):]
            weights[local] = arrays[k]
    NORM_SUFFIXES = ("hc_norm.weight", "q_norm.weight", "k_norm.weight", "q_layernorm.weight", "k_layernorm.weight", "norm_key.weight", "norm_query.weight", "norm_conv.weight")
    def shift_norms(d):
        if args.norm_shift:
            for k in list(d):
                if k.endswith(NORM_SUFFIXES):
                    d[k] = d[k] + args.norm_shift
        return d
    weights = shift_norms(weights)
    qp = quantize_by_keys(g, set(weights), q)
    print("global quantized paths:", sorted(qp), flush=True)
    g.load_weights(list(weights.items()), strict=True)
    g.eval(); mx.eval(g.parameters())

    hidden = mx.tile(g.embed_tokens(input_ids), (1, 1, config.hc_count))
    mx.eval(hidden)
    layer_indices = [int(v) for v in args.layers.split(",")] if args.layers else list(range(config.num_hidden_layers))
    t_all = time.time()
    for li in layer_indices:
        t0 = time.time()
        prefix = f"language_model.model.layers.{li}."
        w = load_prefix(args.model_dir, index, prefix, prefix)
        if args.swap_gate_up:
            sw = {}
            for k, v in w.items():
                if ".gate_proj." in k and "shared_expert_gate" not in k:
                    sw[k.replace(".gate_proj.", ".up_proj.")] = v
                elif ".up_proj." in k:
                    sw[k.replace(".up_proj.", ".gate_proj.")] = v
                else:
                    sw[k] = v
            w = sw
        w = shift_norms(w)
        layer = L.Qwen4ExpDecoderLayer(config, li)
        quantize_by_keys(layer, set(w), q)
        layer.load_weights(list(w.items()), strict=True)
        layer.eval(); mx.eval(layer.parameters())
        is_linear = config.layer_types[li] == "linear_attention"
        cache = L.ArraysCache(size=4) if is_linear else L.QSAKVCache()
        mask = None if is_linear else "causal"
        hidden = layer(hidden, input_ids, mask=mask, cache=cache, position_ids=None)
        mx.eval(hidden)
        print(f"layer {li:2d} ok {'GDN' if is_linear else 'QSA'} {'PLE' if 'ple' in layer else '   '} "
              f"{len(w)} tensors {time.time()-t0:6.1f}s  |h| rms={float(mx.sqrt(mx.mean(hidden.astype(mx.float32)**2))):.4f}", flush=True)
        del layer, cache, w
        gc.collect(); mx.clear_cache()

    reduced = g.hyper_connection_mixer(hidden)
    logits = g.lm_head(reduced)
    mx.eval(reduced, logits)
    print(f"total {time.time()-t_all:.1f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.output), {"input_ids": input_ids, "reduced": reduced, "logits": logits},
                        metadata={"kind": "qb_official_617", "seed": str(config.seed),
                                  "mlx_vlm": mlx_vlm.__version__,
                                  "norm_shift": str(args.norm_shift)})

    # self-consistency: teacher-forced argmax hit rate on the prompt itself
    lf = logits[0].astype(mx.float32)
    lp = lf - mx.logsumexp(lf, axis=-1, keepdims=True)
    am = mx.argmax(lf, axis=-1).tolist()
    hits = sum(int(am[i] == ids[i + 1]) for i in range(len(ids) - 1))
    mean_lp = sum(float(lp[i, ids[i + 1]]) for i in range(len(ids) - 1)) / (len(ids) - 1)
    top = mx.argsort(-lf[-1])[:10].tolist()
    print(f"teacher-forced hits {hits}/{len(ids)-1}  mean logprob {mean_lp:.3f}")
    print("final top10 ids:", top, "logits:", [round(float(lf[-1, t]), 2) for t in top])
    print("argmax per position:", am)


if __name__ == "__main__":
    main()
