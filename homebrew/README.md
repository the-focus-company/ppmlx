# Homebrew Formula for ppmlx

This directory contains the Homebrew formula for ppmlx. To distribute it via
Homebrew, the formula file must live in a separate **tap** repository on GitHub.

## For users: Installing ppmlx via Homebrew

```bash
brew tap PingCompany/ppmlx
brew install ppmlx
```

After installation:

```bash
ppmlx pull llama3      # download a model
ppmlx run llama3       # interactive chat
ppmlx serve            # start the OpenAI-compatible API server
```

## For maintainers: Setting up the tap

1. Create a public repository named **`PingCompany/homebrew-ppmlx`** on GitHub.
2. Copy `Formula/ppmlx.rb` into that repo at the path `Formula/ppmlx.rb`.
3. Push. Users can now `brew tap PingCompany/ppmlx && brew install ppmlx`.

## Updating the formula after a new release

Run the helper script from this repo:

```bash
# Auto-detect the latest version on PyPI
./homebrew/update_formula.sh

# Or pin a specific version
./homebrew/update_formula.sh 0.2.0
```

The script will:
- Fetch the sdist tarball URL from PyPI
- Compute the SHA256 hash
- Patch `Formula/ppmlx.rb` in-place

Then copy the updated formula to the tap repo and push.

## CI automation

The workflow at `.github/workflows/homebrew-update.yml` runs automatically after
each successful Release workflow. It updates the formula and opens a PR on the
tap repo. For this to work, add a **`HOMEBREW_TAP_TOKEN`** secret to the ppmlx
repo — a GitHub PAT with `repo` scope that can push to
`PingCompany/homebrew-ppmlx`.
