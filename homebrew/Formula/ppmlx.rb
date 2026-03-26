class Ppmlx < Formula
  include Language::Python::Virtualenv

  desc "CLI for running LLMs on Apple Silicon via MLX"
  homepage "https://ppmlx.dev"
  url "https://files.pythonhosted.org/packages/source/p/ppmlx/ppmlx-0.1.0.tar.gz"
  # To compute: curl -sL <url> | shasum -a 256
  sha256 "UPDATE_WITH_ACTUAL_SHA256"
  license "MIT"
  head "https://github.com/PingCompany/ppmlx.git", branch: "main"

  depends_on "python@3.11"
  depends_on :macos
  depends_on arch: :arm64

  def install
    virtualenv_create(libexec, "python3.11")

    # Install ppmlx with all dependencies (including optional embeddings)
    # in a single pip invocation. This lets pip handle dependency resolution
    # and avoids maintaining a separate list that drifts from pyproject.toml.
    system libexec/"bin/pip", "install", ".[embeddings]"

    (bin/"ppmlx").write_env_script libexec/"bin/ppmlx", PATH: "#{libexec}/bin:#{ENV["PATH"]}"
  end

  def caveats
    <<~EOS
      ppmlx requires Apple Silicon (M1/M2/M3/M4) and macOS 13+.

      Quick start:
        ppmlx pull llama3
        ppmlx run llama3
        ppmlx serve          # OpenAI-compatible API on :6767
    EOS
  end

  test do
    assert_match "ppmlx", shell_output("#{bin}/ppmlx --help")
  end
end
