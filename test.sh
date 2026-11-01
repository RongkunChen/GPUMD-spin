#!/usr/bin/env bash
# test.sh -- smoke tests on Linux/macOS (UNTESTED on this release)
set -euo pipefail
cd "$(dirname "$0")"
python3 tests/smoke.py --bin src --json smoke_result.json
