# Parity fixtures

This directory is **not** tracked by git. It holds the reference tensors used to
prove that the Swift port computes the same values as the Python reference
implementation.

The fixtures are large (~287 MB) and only useful to someone who already has the
full checkpoint (~100 GB), so they are kept out of the repository. Regenerate
them locally with the scripts in `Scripts/` — see `docs/parity-method.md` for
the method and the exact commands.
