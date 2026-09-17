#!/usr/bin/env zsh

# Download a Hugging Face repository directly, with resumable per-file downloads.
# This is intentionally independent from the Swift URLSession downloader: a lost
# connection on a multi-gigabyte LFS object must not discard the partial shard.

set -euo pipefail

MODEL_ID="${1:-Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP}"
MODELS_DIR="${2:-${QWEN38_MODELS_DIR:-$HOME/models}}"
START_PATH="${3:-}"
END_PATH="${4:-}"
TARGET="${MODELS_DIR}/${MODEL_ID}"
API_URL="https://huggingface.co/api/models/${MODEL_ID}"
BASE_URL="https://huggingface.co/${MODEL_ID}/resolve/main"

if ! command -v curl >/dev/null 2>&1 || ! command -v jq >/dev/null 2>&1; then
  print -u2 "curl et jq sont nécessaires."
  exit 1
fi

TMP_DIR="$(mktemp -d /private/tmp/qwen38-hf-download.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT INT TERM
MANIFEST="${TMP_DIR}/manifest.tsv"
NETRC_FILE="${TMP_DIR}/netrc"
CURL_AUTH_ARGS=()

# Prefer an explicitly supplied token, otherwise reuse the local Hugging Face
# CLI session. The token lives only in a mode-600 temporary netrc file; it is
# never placed in the curl command line or printed to the terminal.
HF_SESSION_TOKEN="${HF_TOKEN:-}"
if [[ "$HF_SESSION_TOKEN" == "" ]] && command -v hf >/dev/null 2>&1; then
  HF_SESSION_TOKEN="$(hf auth token 2>/dev/null || true)"
fi
if [[ "$HF_SESSION_TOKEN" != "" ]]; then
  umask 077
  print -r -- "machine huggingface.co login token password ${HF_SESSION_TOKEN}" > "$NETRC_FILE"
  CURL_AUTH_ARGS=(--netrc-file "$NETRC_FILE")
  print "Session Hugging Face locale utilisée."
else
  print -u2 "Aucun token Hugging Face local trouvé ; téléchargement anonyme."
fi

mkdir -p "$TARGET"

print "Récupération du manifeste ${MODEL_ID}…"
curl --fail --silent --show-error --location \
  --retry 10 --retry-all-errors --retry-delay 3 \
  "${CURL_AUTH_ARGS[@]}" \
  "$API_URL" -o "${TMP_DIR}/model.json"

# Keep only model artifacts. Hidden macOS metadata and repository source files
# are deliberately excluded; the model checkpoint itself is not.
jq -r '
  .siblings[]
  | select(.rfilename != null)
  | select(
      (.rfilename | test("\\.(safetensors|json|jinja|txt|model|vocab)$"))
      or .rfilename == "LICENSE"
      or .rfilename == "README.md"
      or .rfilename == ".gitattributes"
    )
  | [(.rfilename), ((.size // 0) | tostring)]
  | @tsv
' "${TMP_DIR}/model.json" > "$MANIFEST"

FILE_COUNT="$(wc -l < "$MANIFEST" | tr -d ' ')"
if [[ "$FILE_COUNT" == "0" ]]; then
  print -u2 "Manifeste vide pour ${MODEL_ID}."
  exit 1
fi

print "${FILE_COUNT} fichiers à vérifier dans ${TARGET}"

STARTED="${START_PATH:-yes}"
while IFS=$'\t' read -r RELATIVE_PATH EXPECTED_SIZE; do
  if [[ "$START_PATH" != "" && "$STARTED" != "yes" ]]; then
    if [[ "$RELATIVE_PATH" != "$START_PATH" ]]; then
      continue
    fi
    STARTED="yes"
    print "Démarrage demandé à ${START_PATH}"
  fi

  if [[ "$START_PATH" != "" && "$STARTED" != "yes" ]]; then
    continue
  fi

  DEST="${TARGET}/${RELATIVE_PATH}"
  PART="${DEST}.part"
  PARENT="${DEST:h}"
  mkdir -p "$PARENT"

  if [[ -f "$DEST" ]]; then
    ACTUAL_SIZE="$(stat -f '%z' "$DEST")"
    # A final file can only be produced by this script after its post-download
    # validation. Some HF manifests omit .size, so an existing final artifact
    # is still safe to skip when EXPECTED_SIZE is zero.
    if [[ "$EXPECTED_SIZE" == "0" || "$ACTUAL_SIZE" == "$EXPECTED_SIZE" ]]; then
      print "OK   ${RELATIVE_PATH} (${ACTUAL_SIZE} octets)"
      continue
    fi
    print -u2 "Taille incorrecte, reprise : ${RELATIVE_PATH} (${ACTUAL_SIZE}/${EXPECTED_SIZE})"
    mv -f "$DEST" "$PART"
  fi

  CURRENT_SIZE=0
  if [[ -f "$PART" ]]; then
    CURRENT_SIZE="$(stat -f '%z' "$PART")"
  fi
  print "GET  ${RELATIVE_PATH} — reprise à ${CURRENT_SIZE} octets"

  # --continue-at - preserves the partial LFS object after a disconnect.
  curl --fail --location --silent --show-error --progress-bar \
    --retry 20 --retry-all-errors --retry-delay 5 \
    --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
    --continue-at - \
    "${CURL_AUTH_ARGS[@]}" \
    --output "$PART" "${BASE_URL}/${RELATIVE_PATH}"

  ACTUAL_SIZE="$(stat -f '%z' "$PART")"
  if [[ "$EXPECTED_SIZE" != "0" && "$ACTUAL_SIZE" != "$EXPECTED_SIZE" ]]; then
    print -u2 "Taille inattendue après téléchargement de ${RELATIVE_PATH}: ${ACTUAL_SIZE}/${EXPECTED_SIZE}"
    exit 1
  fi
  mv -f "$PART" "$DEST"
  print "DONE ${RELATIVE_PATH} (${ACTUAL_SIZE} octets)"
  if [[ "$END_PATH" != "" && "$RELATIVE_PATH" == "$END_PATH" ]]; then
    print "Plage terminée à ${END_PATH}"
    break
  fi
done < "$MANIFEST"

if [[ "$START_PATH" != "" && "$STARTED" != "yes" ]]; then
  print -u2 "Fichier de départ introuvable dans le manifeste : ${START_PATH}"
  exit 1
fi

print "Téléchargement terminé : ${TARGET}"
