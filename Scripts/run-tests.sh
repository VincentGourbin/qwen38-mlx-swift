#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
derived_data="${QWEN38_DERIVED_DATA:-${repo_root}/.xcodebuild-tests}"

# xcodebuild only forwards test-runner environment variables with the
# TEST_RUNNER_ prefix. MLX's compiled-function/eval lock ordering must stay
# serial while tests exercise the GPU runtime.
export TEST_RUNNER_SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH="1"

exec xcodebuild \
    -scheme Qwen38MLXSwift-Package \
    -configuration Debug \
    -destination 'platform=macOS' \
    -derivedDataPath "${derived_data}" \
    -skipMacroValidation \
    -skipPackageUpdates \
    test
