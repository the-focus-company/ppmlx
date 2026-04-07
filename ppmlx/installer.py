"""Interactive component installer / uninstaller for ppmlx.

Manages optional heavy dependencies (voice, vision, embeddings, analytics)
that are not included in the base install to keep the footprint small.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

console = Console()

_CHECKMARK = "[green]✓[/green]"
_CROSS     = "[red]✗[/red]"


@dataclass
class Component:
    """An optional installable component."""
    key: str
    label: str
    description: str
    packages: list[str]            # pip packages to install
    check_imports: list[str]       # modules to import to verify install
    size_hint: str                 # rough download size (no markup)
    extras_key: str | None = None  # pyproject extras group name
    requires_brew: list[str] = field(default_factory=list)


COMPONENTS: list[Component] = [
    Component(
        key="vision",
        label="Vision / Multimodal",
        description="Image understanding with vision-language models (VLMs)",
        packages=["mlx-vlm>=0.1.18"],
        check_imports=["mlx_vlm"],
        size_hint="~500 MB",
        extras_key="vision",
    ),
    Component(
        key="embeddings",
        label="Embeddings",
        description="Local text embedding models (semantic search, RAG)",
        packages=["mlx-embeddings>=0.0.5"],
        check_imports=["mlx_embeddings"],
        size_hint="~200 MB",
        extras_key="embeddings",
    ),
    Component(
        key="voice",
        label="Voice I/O",
        description="Push-to-talk (Whisper STT) + spoken output (Voxtral TTS)",
        packages=["mlx-whisper>=0.4", "mlx-audio>=0.4", "sounddevice>=0.4", "soundfile>=0.12"],
        check_imports=["mlx_whisper", "mlx_audio", "sounddevice", "soundfile"],
        size_hint="~2 GB (models on first use)",
        extras_key="voice",
        requires_brew=["portaudio"],
    ),
    Component(
        key="analytics",
        label="Analytics",
        description="Anonymous usage statistics (opt-in)",
        packages=["posthog>=7.9,<8"],
        check_imports=["posthog"],
        size_hint="~200 KB",
        extras_key="analytics",
    ),
]

_COMPONENT_MAP = {c.key: c for c in COMPONENTS}


# ── Helpers ────────────────────────────────────────────────────────────

def _is_installed(component: Component) -> bool:
    """Return True if all required imports succeed."""
    for mod in component.check_imports:
        try:
            importlib.import_module(mod.replace("-", "_"))
        except ImportError:
            return False
    return True


def _brew_installed(formula: str) -> bool:
    try:
        r = subprocess.run(["brew", "list", formula], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _run_pip(args: list[str], timeout: int = 300) -> bool:
    cmd = [sys.executable, "-m", "pip"] + args
    console.print(f"  [dim]{' '.join(cmd)}[/dim]")
    return subprocess.run(cmd, timeout=timeout).returncode == 0


# ── Install ────────────────────────────────────────────────────────────

def _install_component(component: Component) -> bool:
    """Install a component's packages. Returns True on success."""
    for formula in component.requires_brew:
        if not _brew_installed(formula):
            console.print(f"  Installing brew dep: [cyan]{formula}[/cyan]")
            if subprocess.run(["brew", "install", formula], timeout=120).returncode != 0:
                console.print(f"  [red]brew install {formula} failed[/red]")
                return False
    return _run_pip(["install", "--quiet"] + component.packages)


def install_component(key: str) -> bool:
    """Install a single component by key. Returns True on success."""
    comp = _COMPONENT_MAP.get(key)
    if not comp:
        console.print(f"[red]Unknown component: {key!r}[/red]")
        console.print(f"Available: {', '.join(_COMPONENT_MAP)}")
        return False
    if _is_installed(comp):
        console.print(f"[green]{comp.label} is already installed.[/green]")
        return True
    console.print(f"Installing [bold]{comp.label}[/bold]…")
    ok = _install_component(comp)
    if ok:
        console.print(f"  {_CHECKMARK} [green]{comp.label} installed[/green]")
    else:
        console.print(f"  {_CROSS} [red]Installation failed.[/red]")
        console.print(f"  [dim]Try manually: pip install {' '.join(comp.packages)}[/dim]")
    return ok


# ── Uninstall ──────────────────────────────────────────────────────────

def _uninstall_component(component: Component) -> bool:
    """Uninstall a component's pip packages. Returns True on success."""
    # Extract bare package names (strip version specifiers)
    pkg_names = [p.split(">=")[0].split("==")[0].split("<")[0].strip()
                 for p in component.packages]
    return _run_pip(["uninstall", "--yes"] + pkg_names)


def uninstall_component(key: str) -> bool:
    """Uninstall a component by key. Returns True on success."""
    comp = _COMPONENT_MAP.get(key)
    if not comp:
        console.print(f"[red]Unknown component: {key!r}[/red]")
        console.print(f"Available: {', '.join(_COMPONENT_MAP)}")
        return False
    if not _is_installed(comp):
        console.print(f"[yellow]{comp.label} is not installed.[/yellow]")
        return True
    console.print(f"Uninstalling [bold]{comp.label}[/bold]…")
    ok = _uninstall_component(comp)
    if ok:
        console.print(f"  {_CHECKMARK} [green]{comp.label} removed[/green]")
    else:
        console.print(f"  {_CROSS} [red]Uninstall failed.[/red]")
    return ok


# ── Status table ───────────────────────────────────────────────────────

def status_table() -> None:
    """Print a table showing which components are installed."""
    table = Table(title="ppmlx components", box=None, padding=(0, 2))
    table.add_column("", width=2)
    table.add_column("Component", style="bold")
    table.add_column("Description", style="dim")
    table.add_column("Size", style="dim", justify="right")
    table.add_column("Status", justify="right")

    for comp in COMPONENTS:
        ok = _is_installed(comp)
        mark = _CHECKMARK if ok else _CROSS
        status = "[green]installed[/green]" if ok else "[dim]not installed[/dim]"
        table.add_row(mark, comp.label, comp.description, comp.size_hint, status)

    console.print(table)
    console.print()
    console.print("[dim]Install:   ppmlx install <component>[/dim]")
    console.print("[dim]Uninstall: ppmlx install --remove <component>[/dim]")


# ── Interactive wizards ────────────────────────────────────────────────

def _q_style():
    import questionary
    return questionary.Style([
        ("checkbox-selected", "fg:ansigreen bold"),
        ("selected",          "fg:ansigreen"),
        ("pointer",           "fg:ansicyan bold"),
        ("highlighted",       "fg:ansicyan"),
        ("answer",            "fg:ansigreen bold"),
        ("question",          "bold"),
    ])


def install_interactive() -> None:
    """Interactive install wizard — checkbox picker for not-yet-installed components."""
    import questionary

    console.print()
    console.print("[bold]ppmlx component installer[/bold]")
    console.print("[dim]Base install is lean (~400 MB). Add components as needed.[/dim]")
    console.print()

    # Show status table first
    status_table()

    installable = [c for c in COMPONENTS if not _is_installed(c)]
    if not installable:
        console.print("[green]All components are already installed.[/green]")
        return

    choices = [
        questionary.Choice(
            # Plain text — no Rich markup; questionary uses its own renderer
            title=f"{comp.label}  ({comp.size_hint})  —  {comp.description}",
            value=comp.key,
        )
        for comp in installable
    ]

    selected_keys: list[str] | None = questionary.checkbox(
        "Select components to install:",
        choices=choices,
        style=_q_style(),
    ).ask()

    if not selected_keys:
        console.print("[dim]Nothing selected.[/dim]")
        return

    for key in selected_keys:
        comp = _COMPONENT_MAP[key]
        console.print(f"\nInstalling [bold]{comp.label}[/bold]…")
        ok = _install_component(comp)
        if ok:
            console.print(f"  {_CHECKMARK} [green]{comp.label} installed[/green]")
        else:
            console.print(f"  {_CROSS} [red]{comp.label} failed[/red]")
            console.print(f"  [dim]Try manually: pip install {' '.join(comp.packages)}[/dim]")

    console.print("\n[green]Done.[/green] Run [bold]ppmlx install --status[/bold] to verify.")


def uninstall_interactive() -> None:
    """Interactive uninstall wizard — checkbox picker for installed components."""
    import questionary

    console.print()
    console.print("[bold]ppmlx component uninstaller[/bold]")
    console.print()

    installed = [c for c in COMPONENTS if _is_installed(c)]
    if not installed:
        console.print("[dim]No optional components are installed.[/dim]")
        return

    choices = [
        questionary.Choice(
            title=f"{comp.label}  ({comp.size_hint})  —  {comp.description}",
            value=comp.key,
        )
        for comp in installed
    ]

    selected_keys: list[str] | None = questionary.checkbox(
        "Select components to remove:",
        choices=choices,
        style=_q_style(),
    ).ask()

    if not selected_keys:
        console.print("[dim]Nothing selected.[/dim]")
        return

    # Confirm
    labels = ", ".join(_COMPONENT_MAP[k].label for k in selected_keys)
    confirmed = questionary.confirm(
        f"Remove {labels}?", default=False,
    ).ask()
    if not confirmed:
        console.print("[dim]Cancelled.[/dim]")
        return

    for key in selected_keys:
        comp = _COMPONENT_MAP[key]
        console.print(f"\nRemoving [bold]{comp.label}[/bold]…")
        ok = _uninstall_component(comp)
        if ok:
            console.print(f"  {_CHECKMARK} [green]{comp.label} removed[/green]")
        else:
            console.print(f"  {_CROSS} [red]Removal failed[/red]")

    console.print("\n[green]Done.[/green]")


# ── On-demand prompt ───────────────────────────────────────────────────

def prompt_install_if_missing(key: str, feature_name: str) -> bool:
    """If a component is missing, offer to install it interactively.

    Returns True if available after the prompt.
    """
    comp = _COMPONENT_MAP.get(key)
    if not comp:
        return False
    if _is_installed(comp):
        return True

    import questionary
    console.print(
        f"\n[yellow]{feature_name} requires the "
        f"[bold]{comp.label}[/bold] component ({comp.size_hint}).[/yellow]"
    )
    console.print(f"[dim]{comp.description}[/dim]\n")

    if not questionary.confirm(f"Install {comp.label} now?", default=True).ask():
        console.print(f"[dim]Skipped. Run: ppmlx install {key}[/dim]")
        return False

    return install_component(key)
