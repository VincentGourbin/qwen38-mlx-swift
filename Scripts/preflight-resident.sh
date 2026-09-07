#!/usr/bin/env bash
# Préflight obligatoire avant tout run Flash-Next résident (PLAN.md §6.0 rév. 4,
# RÉPONSE du 2026-09-07). Sortie 0 = feu vert, 1 = refus.
#
# La bonne métrique n'est PAS « PhysMem used » de top (qui compte le cache de
# fichiers, ~68 Go après une lecture du checkpoint) mais la mémoire anonyme
# réellement occupée par les autres processus : pages anonymes + pages stockées
# dans le compresseur + swap utilisé. Budget : 96 Go − ~77 Go (process résident
# mesuré V54) − ~4,5 Go (noyau wired) − ~4 Go de marge ≈ 10 Go.
set -euo pipefail

limit_gb="${QWEN38_PREFLIGHT_LIMIT_GB:-9}"
checkpoint="${1:-/Volumes/Lexar/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP}"
page=$(vm_stat | head -1 | grep -o '[0-9]*')
anon=$(vm_stat | awk '/Anonymous pages/ {gsub("\\.","",$3); print $3}')
comp=$(vm_stat | awk '/Pages stored in compressor/ {gsub("\\.","",$5); print $5}')
swap_mb=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9]+)[,.]([0-9]+)M.*/\1/')
anon_gb=$(python3 -c "print(round($anon*$page/2**30,1))")
comp_gb=$(python3 -c "print(round($comp*$page/2**30,1))")
swap_gb=$(python3 -c "print(round($swap_mb/1024,1))")
total_gb=$(python3 -c "print(round($anon_gb+$comp_gb+$swap_gb,1))")
wired_mb=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo "?")
gpu_util=$(ioreg -r -c AGXAccelerator -d 1 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)
other=$(pgrep -fl 'qwen38|qwen38-bench-ui|mlx_vlm|python' | grep -v preflight | wc -l | tr -d ' ')

echo "mémoire anonyme        : ${anon_gb} Go"
echo "compresseur (stocké)   : ${comp_gb} Go"
echo "swap utilisé           : ${swap_gb} Go"
echo "total à évincer        : ${total_gb} Go  (seuil ${limit_gb} Go)"
echo "iogpu.wired_limit_mb   : ${wired_mb}  (0 = défaut ≈ 72 Go de working set)"
echo "GPU utilisation        : ${gpu_util:-?} %"
echo "process qwen38/python  : ${other}"
status=0
if [ ! -f "${checkpoint}/config.json" ]; then
  echo "REFUS : checkpoint absent (Lexar non monté ?) : ${checkpoint}"; status=1
fi
if python3 -c "import sys; sys.exit(0 if $total_gb <= $limit_gb else 1)"; then
  echo "OK : marge mémoire suffisante"
else
  echo "REFUS : fermer Arc/Teams/ChatGPT/Xcode/sessions Claude surnuméraires puis relancer"
  ps -axo rss,comm | sort -rn | head -8 | awk '{printf "  %.1f Go  %s\n",$1/1048576,$2}'
  status=1
fi
exit $status
