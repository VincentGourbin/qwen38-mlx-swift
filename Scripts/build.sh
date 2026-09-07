#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
derived_data="${QWEN38_DERIVED_DATA:-${repo_root}/.xcodebuild}"
# Build the package scheme so both the CLI and the GUI are refreshed.  The
# GUI-only product scheme leaves qwen38 stale, which is particularly easy to
# miss when adding a diagnostic command such as `mtp-probe`.
scheme="${QWEN38_SCHEME:-Qwen38MLXSwift-Package}"
configuration="${QWEN38_CONFIGURATION:-Debug}"

# MLX inference needs the bundle produced by Xcode (not just a SwiftPM
# executable): mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib.
exec xcodebuild \
    -scheme "${scheme}" \
    -configuration "${configuration}" \
    -destination 'platform=macOS,arch=arm64' \
    -derivedDataPath "${derived_data}" \
    -skipMacroValidation \
    -skipPackageUpdates \
    build
