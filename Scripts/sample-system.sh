#!/usr/bin/env bash
# Échantillonneur système à lancer EN PARALLÈLE d'un run résident (P1) :
#   Scripts/sample-system.sh results/p1-system.tsv &
# Toutes les 2 s : mémoire anonyme, compresseur, décompressions cumulées,
# swap, GPU %, RSS du process qwen38. Les colonnes « decompressions » et
# « pageins » sont cumulatives : leur delta par intervalle mesure le thrash.
set -uo pipefail
out="${1:-results/system-samples.tsv}"
interval="${2:-2}"
mkdir -p "$(dirname "$out")"
page=$(vm_stat | head -1 | grep -o '[0-9]*')
echo -e "time\tanon_gb\tcompressor_gb\tdecompressions\tpageins\tswap_used_mb\tgpu_pct\tqwen38_rss_gb" > "$out"
while true; do
  vs=$(vm_stat)
  anon=$(echo "$vs" | awk '/Anonymous pages/ {gsub("\\.","",$3); print $3}')
  comp=$(echo "$vs" | awk '/Pages stored in compressor/ {gsub("\\.","",$5); print $5}')
  dec=$(echo "$vs" | awk '/Decompressions/ {gsub("\\.","",$2); print $2}')
  pin=$(echo "$vs" | awk '/^Pageins/ {gsub("\\.","",$2); print $2}')
  swap=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9]+)[,.]([0-9]+)M.*/\1/')
  gpu=$(ioreg -r -c AGXAccelerator -d 1 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)
  rss=$(ps -axo rss,comm | awk '$2 ~ /qwen38/ {s+=$1} END {printf "%.1f", s/1048576}')
  printf "%s\t%.1f\t%.1f\t%s\t%s\t%s\t%s\t%s\n" "$(date +%H:%M:%S)" \
    "$(python3 -c "print($anon*$page/2**30)")" "$(python3 -c "print($comp*$page/2**30)")" \
    "$dec" "$pin" "$swap" "${gpu:-?}" "$rss" >> "$out"
  sleep "$interval"
done
