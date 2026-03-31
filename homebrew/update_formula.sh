#!/usr/bin/env bash
# update_formula.sh — Fetch latest ppmlx release from PyPI and update the
# Homebrew formula with the correct version and SHA256 hash.
#
# Usage:
#   ./homebrew/update_formula.sh            # auto-detect latest version
#   ./homebrew/update_formula.sh 0.2.0      # pin to a specific version
set -euo pipefail

PACKAGE="ppmlx"
FORMULA="$(cd "$(dirname "$0")" && pwd)/Formula/ppmlx.rb"

# ── resolve version ──────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
  VERSION="$1"
else
  echo "Fetching latest version from PyPI..."
  VERSION=$(curl -fsSL "https://pypi.org/pypi/${PACKAGE}/json" | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])")
fi

echo "Version: ${VERSION}"

# ── build the sdist URL and fetch SHA256 ─────────────────────────────────────
SDIST_URL="https://files.pythonhosted.org/packages/source/p/${PACKAGE}/${PACKAGE}-${VERSION}.tar.gz"

echo "Downloading sdist to compute SHA256..."
if command -v sha256sum &>/dev/null; then
  SHA256=$(curl -fsSL "${SDIST_URL}" | sha256sum | awk '{print $1}')
else
  SHA256=$(curl -fsSL "${SDIST_URL}" | shasum -a 256 | awk '{print $1}')
fi
echo "SHA256:  ${SHA256}"

if [[ -z "${SHA256}" || "${#SHA256}" -ne 64 ]]; then
  echo "ERROR: failed to compute SHA256 — check that version ${VERSION} exists on PyPI." >&2
  exit 1
fi

# ── update the formula ───────────────────────────────────────────────────────
if [[ ! -f "${FORMULA}" ]]; then
  echo "ERROR: formula not found at ${FORMULA}" >&2
  exit 1
fi

# Portable in-place sed (macOS uses -i '', GNU sed uses -i)
_sed_i() {
  if sed --version 2>/dev/null | grep -q GNU; then
    sed -i "$@"
  else
    sed -i '' "$@"
  fi
}

# Replace the url line
_sed_i "s|url \"https://files.pythonhosted.org/packages/source/p/ppmlx/ppmlx-.*\.tar\.gz\"|url \"${SDIST_URL}\"|" "${FORMULA}"

# Replace the sha256 line (matches hex strings and placeholder text)
_sed_i "s|sha256 \"[a-zA-Z0-9_]*\"|sha256 \"${SHA256}\"|" "${FORMULA}"

echo ""
echo "Formula updated:"
echo "  url:    ${SDIST_URL}"
echo "  sha256: ${SHA256}"
echo ""
echo "Next steps:"
echo "  1. Review the diff:   git diff homebrew/Formula/ppmlx.rb"
echo "  2. Commit and push to the tap repo: PingCompany/homebrew-ppmlx"
