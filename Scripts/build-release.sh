#!/usr/bin/env bash
set -euo pipefail

# Wrapper autour de Scripts/build.sh qui force la configuration Release.
# Les probes (flash-generate-probe, flash-chat-probe), le bench
# flash-layer-bench, les runs H6 et la démo G-8 doivent tourner sur ce
# binaire (coût hôte MLX divisé par ~1,2-1,8 par rapport à Debug, cf.
# docs/knowledge/log.md « P0 rejoué en Release »). Debug (Scripts/build.sh
# sans variable) reste la configuration des tests (Scripts/run-tests.sh).

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QWEN38_CONFIGURATION=Release
exec "${repo_root}/Scripts/build.sh"
