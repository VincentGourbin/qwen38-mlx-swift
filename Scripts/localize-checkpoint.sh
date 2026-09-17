#!/usr/bin/env bash
# Crée sur le SSD interne une vue « hybride » d'un checkpoint Flash-Next posé
# sur un volume externe : les shards de la table n-gram (lus par petits accès aléatoires à
# chaque token, ceux qui souffrent le plus de l'USB) sont COPIÉS, les autres
# shards (lus une fois, séquentiellement, au chargement) sont des LIENS
# SYMBOLIQUES vers le volume externe, et les petits fichiers (config, tokenizer…) sont
# copiés. Le loader suit les liens sans rien savoir. Avec --full, tout est
# copié (plus aucune dépendance au volume externe). Idempotent : relancer complète.
#
#   Scripts/localize-checkpoint.sh <src volume externe> <dst SSD interne> [--full]
set -euo pipefail
export LC_ALL=C
src="${1:?src}"; dst="${2:?dst}"; mode="${3:-ngram}"
[ -f "$src/model.safetensors.index.json" ] || { echo "index absent : $src"; exit 1; }
mkdir -p "$dst"
python3 - "$src" "$dst" "$mode" <<'EOF'
import json, os, shutil, subprocess, sys, time
src, dst, mode = sys.argv[1:4]
idx = json.load(open(os.path.join(src, 'model.safetensors.index.json')))['weight_map']
shards = sorted(set(idx.values()))
ngram = {idx[k] for k in idx if 'ngram' in k}
to_copy = shards if mode == '--full' else sorted(ngram)
to_link = [] if mode == '--full' else [s for s in shards if s not in ngram]
small = [f for f in os.listdir(src) if not f.endswith('.safetensors') and not f.startswith('.')]
for f in small:
    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
for s in to_link:
    p = os.path.join(dst, s)
    if os.path.islink(p) or os.path.exists(p):
        if os.path.islink(p): os.unlink(p)
        else: continue  # déjà une vraie copie, on la garde
    os.symlink(os.path.join(src, s), p)
free = shutil.disk_usage(dst).free
need = sum(os.path.getsize(os.path.join(src, s)) for s in to_copy
           if not (os.path.exists(os.path.join(dst, s)) and not os.path.islink(os.path.join(dst, s))
                   and os.path.getsize(os.path.join(dst, s)) == os.path.getsize(os.path.join(src, s))))
print(f"à copier : {need/1e9:.1f} Go · libre : {free/1e9:.1f} Go · liens : {len(to_link)} · copies : {len(to_copy)}")
if need > free - 15e9:
    print("REFUS : il resterait moins de 15 Go libres sur le SSD"); sys.exit(2)
t0 = time.time(); done = 0
for s in to_copy:
    p = os.path.join(dst, s); q = os.path.join(src, s)
    if os.path.islink(p): os.unlink(p)
    if os.path.exists(p) and os.path.getsize(p) == os.path.getsize(q):
        continue
    subprocess.run(['cp', q, p + '.part'], check=True)
    os.rename(p + '.part', p); done += os.path.getsize(q)
    print(f"  {s} copié ({done/1e9:.1f} Go, {time.time()-t0:.0f} s)", flush=True)
# vérification : chaque shard de l'index est résolvable et a la bonne taille
bad = [s for s in shards if not os.path.exists(os.path.join(dst, s)) or os.path.getsize(os.path.join(dst, s)) != os.path.getsize(os.path.join(src, s))]
print("vérification :", "OK" if not bad else f"MANQUANTS {bad}")
sys.exit(1 if bad else 0)
EOF
