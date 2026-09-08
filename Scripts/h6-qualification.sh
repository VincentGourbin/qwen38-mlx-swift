#!/usr/bin/env bash
# H6 — qualification élargie Flash-Next en UN SEUL process serveur (PLAN.md
# §6.2 H6, RÉPONSE P1 du 2026-09-08). Charge le modèle une fois (~90 s), puis
# cinq requêtes OpenAI non-stream, greedy (temperature 0). Résultats dans
# results/flash-qualification-rev4.tsv + JSON bruts dans results/h6/.
#
#   Scripts/h6-qualification.sh [model_dir] [port]
set -uo pipefail
export LC_ALL=C
cd "$(dirname "${BASH_SOURCE[0]}")/.."
model_dir="${1:-/Volumes/Lexar/models/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP}"
port="${2:-8848}"
model_id="$(basename "$model_dir")"
bin=./.xcodebuild/Build/Products/Release/qwen38
image=/Users/vincent/Downloads/licensed-image-2.jpeg
out=results/flash-qualification-rev4.tsv
mkdir -p results/h6
ref="Explique en français qui est le président de la Chine et quel est son rôle."

echo "$(date +%T) démarrage serveur ($model_id, port $port)"
caffeinate -dimsu "$bin" serve --model-path "$model_dir" --port "$port" > results/h6/server.log 2>&1 &
server_pid=$!
trap 'echo "$(date +%T) arrêt serveur"; kill -INT $server_pid 2>/dev/null; sleep 3; kill -TERM $server_pid 2>/dev/null' EXIT
for i in $(seq 1 60); do curl -sf "http://127.0.0.1:$port/healthz" > /dev/null && break; sleep 2; done
curl -s "http://127.0.0.1:$port/healthz"; echo

echo -e "id\tmode\tmax_tokens\tduree_s\tfinish\treasoning_chars\tcontent" > "$out"

# $1 id, $2 mode label, $3 max_tokens, $4 JSON body (sans model)
ask() {
  local id="$1" mode="$2" max="$3" body="$4" t0 t1 dur resp finish content reasoning
  echo "$(date +%T) → $id ($mode, $max tokens)"
  t0=$(date +%s.%N)
  resp=$(curl -s --max-time 1800 "http://127.0.0.1:$port/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model_id\",\"stream\":false,\"temperature\":0,\"max_tokens\":$max,\"mtp\":false,$body}")
  t1=$(date +%s.%N)
  dur=$(python3 -c "print(round($t1-$t0,1))")
  echo "$resp" > "results/h6/$id.json"
  finish=$(python3 -c "import json,sys; r=json.load(open('results/h6/$id.json')); print(r['choices'][0].get('finish_reason'))" 2>/dev/null || echo "ERREUR")
  content=$(python3 -c "import json; r=json.load(open('results/h6/$id.json')); print((r['choices'][0]['message'].get('content') or '').replace('\t',' ').replace('\n',' ⏎ '))" 2>/dev/null || echo "$resp" | head -c 300)
  reasoning=$(python3 -c "import json; r=json.load(open('results/h6/$id.json')); print(len(r['choices'][0]['message'].get('reasoning_content') or ''))" 2>/dev/null || echo 0)
  echo -e "$id\t$mode\t$max\t$dur\t$finish\t$reasoning\t$content" >> "$out"
  echo "   ${dur}s · finish=$finish · reasoning=${reasoning} chars · $content" | cut -c1-300
}

user() { python3 -c "import json,sys; print(json.dumps({'role':'user','content':sys.argv[1]}, ensure_ascii=False))" "$1"; }

# H6.1 — thinking, 200 tokens : </think> doit être fermé (reasoning_content ET content non vides)
ask H6.1 thinking 200 "\"enable_thinking\":true,\"messages\":[$(user "$ref")]"
# H6.2 — quatre prompts, 48 tokens, sans thinking
ask H6.2a instruct 48 "\"enable_thinking\":false,\"messages\":[$(user "Explique en une phrase ce qu'est la photosynthèse.")]"
ask H6.2b instruct 48 "\"enable_thinking\":false,\"messages\":[$(user "Écris une fonction Swift qui inverse une chaîne.")]"
ask H6.2c instruct 48 "\"enable_thinking\":false,\"messages\":[$(user "Quelle est la capitale de l'Australie et pourquoi pas Sydney ?")]"
ask H6.2d instruct 48 "\"enable_thinking\":false,\"messages\":[$(user "Traduis en anglais : Le chat dort sur le canapé.")]"
# H6.3 — image de référence, 48 tokens
b64=$(base64 -i "$image" | tr -d '\n')
ask H6.3 image 48 "\"enable_thinking\":false,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Qui est sur cette image et quel est son rôle ?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$b64\"}}]}]"
# H6.4 — deux tours (historique OpenAI complet), 48 tokens
turn1=$(python3 -c "import json; r=json.load(open('results/h6/H6.2a.json')); print(r['choices'][0]['message'].get('content') or '')" 2>/dev/null)
ask H6.4-t1 instruct 48 "\"enable_thinking\":false,\"messages\":[$(user "$ref")]"
prev=$(python3 -c "import json; r=json.load(open('results/h6/H6.4-t1.json')); print(json.dumps({'role':'assistant','content':r['choices'][0]['message'].get('content') or ''}, ensure_ascii=False))")
ask H6.4-t2 instruct 48 "\"enable_thinking\":false,\"messages\":[$(user "$ref"),$prev,$(user "Et son prédécesseur ?")]"

echo "$(date +%T) terminé — $out"
column -t -s $'\t' "$out" | cut -c1-220
