#!/usr/bin/env python3
"""
ollama_cli_v3.py — Agentic Ollama CLI
Features: auto tool-calling, iterative debug loop, /plan executor, long-term memory

Install deps:
    pip install rich prompt_toolkit ollama requests beautifulsoup4

Run:
    python ollama_cli_v3.py
    python ollama_cli_v3.py --model mistral:7b
    python ollama_cli_v3.py --auto
    python ollama_cli_v3.py --compare
"""

import json, os, sys, subprocess, datetime, argparse, time, re
from pathlib import Path

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.rule import Rule
    from rich import box
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.application import Application
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style as PTStyle
    import ollama
except ImportError:
    print("Missing deps. Run: pip install rich prompt_toolkit ollama")
    sys.exit(1)

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_ENABLED = True
except ImportError:
    WEB_ENABLED = False


CONFIG_PATH  = Path.home() / ".ollama_cli_config.json"
HISTORY_FILE = Path.home() / ".ollama_cli_history"
SAVES_DIR    = Path.home() / ".ollama_cli_saves"
PERSONAS_DIR = Path.home() / ".ollama_cli_personas"
MEMORY_FILE  = Path.home() / ".ollama_cli_memory.json"

for d in [SAVES_DIR, PERSONAS_DIR]:
    d.mkdir(exist_ok=True)

COLORS = {
    "user": "cyan", "ai": "green", "tool": "yellow",
    "err": "bold red", "info": "dim white",
    "rule": "bright_black", "banner": "bright_black",
}

DEFAULT_CONFIG = {
    "model": "llama3.1:8b",
    "show_tokens": False,
    "auto_mode": False,
    "max_auto_steps": 10,
    "max_debug_iters": 5,
    "compare_models": [],
}
# ── Suggested models for first-run picker ────────────────────────────────────

SUGGESTED_MODELS = [
    ("llama3.1:8b",        "Meta Llama 3.1  · 8B  · Great all-rounder         · ~5GB"),
    ("mistral:7b",         "Mistral         · 7B  · Fast, strong reasoning     · ~4GB"),
    ("qwen2.5:7b",         "Qwen 2.5        · 7B  · Excellent for code         · ~5GB"),
    ("qwen2.5:14b",        "Qwen 2.5        · 14B · More capable, needs RAM    · ~9GB"),
    ("llava:7b",           "LLaVA           · 7B  · Multimodal (vision+text)   · ~5GB"),
    ("phi3:mini",          "Phi-3 Mini      · 3B  · Very fast, low RAM         · ~2GB"),
    ("deepseek-coder:6.7b","DeepSeek Coder  · 6.7B· Specialized for code       · ~4GB"),
]

# ── Full model catalogue for /install ────────────────────────────────────────

INSTALLABLE_MODELS = [
    # General purpose
    ("llama3.1:8b",         "Meta Llama 3.1 8B       · Great all-rounder                · ~5GB"),
    ("llama3.1:70b",        "Meta Llama 3.1 70B      · Very capable, needs lots of RAM  · ~40GB"),
    ("llama3.2:3b",         "Meta Llama 3.2 3B       · Tiny and fast                    · ~2GB"),
    ("mistral:7b",          "Mistral 7B               · Fast, strong reasoning            · ~4GB"),
    ("mistral-nemo:12b",    "Mistral Nemo 12B         · Strong multilingual               · ~7GB"),
    ("mixtral:8x7b",        "Mixtral 8x7B             · MoE, very capable                · ~26GB"),
    # Code
    ("qwen2.5:7b",          "Qwen 2.5 7B              · Excellent for code                · ~5GB"),
    ("qwen2.5:14b",         "Qwen 2.5 14B             · More capable                      · ~9GB"),
    ("qwen2.5:32b",         "Qwen 2.5 32B             · High capability                   · ~20GB"),
    ("deepseek-coder:6.7b", "DeepSeek Coder 6.7B      · Specialized for code              · ~4GB"),
    ("deepseek-coder-v2",   "DeepSeek Coder V2        · State of art coding               · ~9GB"),
    ("codellama:7b",        "Code Llama 7B            · Meta's code model                 · ~4GB"),
    ("codellama:13b",       "Code Llama 13B           · Larger code model                 · ~8GB"),
    # Vision / Multimodal
    ("llava:7b",            "LLaVA 7B                 · Vision + text                     · ~5GB"),
    ("llava:13b",           "LLaVA 13B                · Larger vision model               · ~8GB"),
    ("llava-llama3:8b",     "LLaVA Llama3 8B          · Better vision on Llama3           · ~5GB"),
    # Small / Fast
    ("phi3:mini",           "Phi-3 Mini               · Very fast, low RAM                · ~2GB"),
    ("phi3:medium",         "Phi-3 Medium             · Balanced size/quality             · ~8GB"),
    ("gemma2:2b",           "Gemma 2 2B               · Google, very small                · ~2GB"),
    ("gemma2:9b",           "Gemma 2 9B               · Google, solid quality             · ~6GB"),
    ("gemma2:27b",          "Gemma 2 27B              · Google, high quality              · ~16GB"),
    # Reasoning / Specialized  
    ("deepseek-r1:7b",      "DeepSeek R1 7B           · Strong reasoning model            · ~5GB"),
    ("deepseek-r1:14b",     "DeepSeek R1 14B          · Better reasoning                  · ~9GB"),
    ("qwq:32b",             "QwQ 32B                  · Reasoning specialist              · ~20GB"),
    ("command-r:35b",       "Command R 35B            · Cohere, great for RAG             · ~20GB"),

]

# ── Bootstrap helpers ─────────────────────────────────────────────────────────

def _is_ollama_installed():
    return subprocess.run(["which", "ollama"], capture_output=True).returncode == 0

def _is_ollama_running():
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False

def _install_ollama(console):
    import platform
    system = platform.system()
    console.print("\n[bold yellow]Ollama not found.[/bold yellow] Installing automatically...\n")

    if system == "Darwin":
        console.print("[dim]Downloading Ollama for macOS...[/dim]")
        try:
            dest = Path.home() / "Downloads" / "Ollama-darwin.zip"
            subprocess.run(["curl", "-L", "-o", str(dest),
                "https://ollama.com/download/Ollama-darwin.zip"], check=True)
            subprocess.run(["unzip", "-o", str(dest), "-d",
                str(Path.home() / "Downloads")], check=True)
            console.print(
                "\n[bold green]✓ Ollama downloaded![/bold green]\n"
                f"[dim]Open [cyan]{Path.home()}/Downloads/Ollama.app[/cyan] "
                "to complete install, then re-run this CLI.[/dim]"
            )
            return False
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]\n"
                "[dim]Visit [cyan]https://ollama.com/download[/cyan] to install manually.[/dim]")
            return False

    elif system == "Linux":
        console.print("[dim]Running official Ollama Linux installer...[/dim]")
        try:
            subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
                shell=True, check=True)
            console.print("[bold green]✓ Ollama installed![/bold green]")
            return True
        except Exception as e:
            console.print(f"[red]Install failed: {e}[/red]\n"
                "[dim]Visit [cyan]https://ollama.com/download[/cyan] to install manually.[/dim]")
            return False

    elif system == "Windows":
        console.print(
            "[yellow]Windows detected.[/yellow] Download Ollama from:\n"
            "[cyan]https://ollama.com/download/OllamaSetup.exe[/cyan]\n"
            "Run the installer, then re-launch this CLI."
        )
        return False

    else:
        console.print(f"[red]Unsupported platform: {system}[/red]")
        return False

def _start_ollama(console):
    console.print("[dim]Starting Ollama server...[/dim]")
    try:
        subprocess.Popen(["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(16):
            time.sleep(0.5)
            if _is_ollama_running():
                console.print("[bold green]✓ Ollama server started.[/bold green]")
                return True
        console.print("[red]✗ Ollama server did not start in time. Try `ollama serve` manually.[/red]")
        return False
    except FileNotFoundError:
        console.print("[red]✗ Could not start Ollama — binary not found.[/red]")
        return False

def _arrow_select(title, models_list, confirm_text="Confirm"):
    """
    Pure keyboard arrow-key picker.
    ↑↓ to move  · Enter to confirm  · Esc/q to cancel
    """
    items = list(models_list)
    state = {"idx": 0, "result": None, "done": False}

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        state["idx"] = (state["idx"] - 1) % len(items)
        event.app.invalidate()

    @kb.add("down")
    def _down(event):
        state["idx"] = (state["idx"] + 1) % len(items)
        event.app.invalidate()

    @kb.add("enter")
    def _enter(event):
        state["result"] = items[state["idx"]][0]
        state["done"]   = True
        event.app.exit()

    @kb.add("escape")
    @kb.add("q")
    @kb.add("c-c")
    def _cancel(event):
        state["result"] = None
        state["done"]   = True
        event.app.exit()

    def get_content():
        lines = []
        lines.append(("class:title",   f" {title}\n"))
        lines.append(("class:hint",    f" ↑↓ Navigate · Enter = {confirm_text} · Esc = Cancel\n\n"))
        for i, (name, desc) in enumerate(items):
            if i == state["idx"]:
                lines.append(("class:selected", f" ▶  {name:<28} {desc}\n"))
            else:
                lines.append(("class:normal",   f"    {name:<28} {desc}\n"))
        return lines

    style = PTStyle.from_dict({
        "title":    "bold ansiyellow",
        "hint":     "ansigray",
        "selected": "bold ansigreen reverse",
        "normal":   "ansiwhite",
    })

    layout = Layout(Window(content=FormattedTextControl(get_content, focusable=True)))
    app    = Application(layout=layout, key_bindings=kb, style=style, full_screen=False, mouse_support=False)

    try:
        app.run()
    except (KeyboardInterrupt, EOFError):
        return None

    return state["result"]

def _do_pull(console, model_name, config, set_as_default=True):
    """Pull a model with live progress output."""
    console.print(f"\n[bold]Pulling [cyan]{model_name}[/cyan]...[/bold] [dim](this may take a few minutes)[/dim]\n")
    try:
        proc = subprocess.Popen(["ollama", "pull", model_name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            line = line.strip()
            if line:
                console.print(f"[dim]  {line}[/dim]")
        proc.wait()
        if proc.returncode == 0:
            console.print(f"\n[bold green]✓ {model_name} ready![/bold green]\n")
            if set_as_default:
                config["model"] = model_name
                save_config(config)
            return True
        else:
            console.print("[red]✗ Pull failed. Check the model name and try again.[/red]")
            return False
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False

def _pull_model_interactive(console, config):
    """First-run model picker using arrow keys."""
    console.print(Panel(
        "[bold yellow]No models found![/bold yellow]\n"
        "[dim]Let\'s download one to get started.[/dim]",
        border_style="yellow", box=box.ROUNDED
    ))
    console.print()
    model_name = _arrow_select("Choose a model to download", SUGGESTED_MODELS, confirm_text="Download")
    if not model_name:
        return None
    ok = _do_pull(console, model_name, config, set_as_default=True)
    return model_name if ok else None

def bootstrap(console, config):
    """Check Ollama installed → running → has models. Fix each automatically."""
    if not _is_ollama_installed():
        ok = _install_ollama(console)
        if not ok:
            return False

    if not _is_ollama_running():
        ok = _start_ollama(console)
        if not ok:
            return False

    try:
        import ollama as _ol
        raw    = _ol.list()
        models = raw.models if hasattr(raw, "models") else raw.get("models", [])
    except Exception:
        models = []

    if not models:
        model = _pull_model_interactive(console, config)
        if not model:
            console.print("[yellow]No model selected. Pull one later with: ollama pull <model>[/yellow]")

    return True


def load_config():
    if CONFIG_PATH.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text())}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

def load_memory():
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            pass
    return []

def save_memory(memories):
    MEMORY_FILE.write_text(json.dumps(memories, indent=2))

HELP_TEXT = (
    "[bold yellow]Chat[/bold yellow]\n"
    "  Just type to chat. Markdown, code blocks, tables all rendered.\n\n"
    "[bold yellow]Agentic Features (NEW in v3)[/bold yellow]\n"
    "  [cyan]/auto[/cyan]                    Toggle autonomous tool-calling mode\n"
    "  [cyan]/plan[/cyan] [dim]<goal>[/dim]             Break goal into steps and execute\n"
    "  [cyan]/run[/cyan] [dim]<file.py>[/dim]           Run code, auto-fix errors in a loop\n"
    "  [cyan]/remember[/cyan] [dim]<fact>[/dim]         Store a fact in long-term memory\n"
    "  [cyan]/memories[/cyan]                List all stored memories\n"
    "  [cyan]/forget[/cyan] [dim]<id>[/dim]             Delete a memory by ID\n\n"
    "[bold yellow]Commands[/bold yellow]\n"
    "  [cyan]/help[/cyan]                    This help\n"
    "  [cyan]/model[/cyan]                   Switch active model (arrow-key picker)\n"
    "  [cyan]/current[/cyan]                 Show the currently active model\n"
    "  [cyan]/install[/cyan]                 Browse & install models from full catalogue\n"
    "  [cyan]/models[/cyan]                  List local Ollama models\n"
    "  [cyan]/compare[/cyan] [dim]<m1> <m2>[/dim]       Side-by-side model comparison\n"
    "  [cyan]/system[/cyan] [dim]<prompt>[/dim]          Set/view system prompt\n"
    "  [cyan]/persona[/cyan] [dim]<n>[/dim]           Load saved persona\n"
    "  [cyan]/personas[/cyan]                List saved personas\n"
    "  [cyan]/save-persona[/cyan] [dim]<n>[/dim]      Save current system prompt as persona\n"
    "  [cyan]/clear[/cyan]                   Clear conversation\n"
    "  [cyan]/retry[/cyan]                   Regenerate last response\n"
    "  [cyan]/save[/cyan] [dim]<n>[/dim]             Save conversation\n"
    "  [cyan]/load[/cyan] [dim]<n>[/dim]             Load conversation\n"
    "  [cyan]/list[/cyan]                    List saved conversations\n"
    "  [cyan]/info[/cyan]                    Session info\n"
    "  [cyan]/tokens[/cyan]                  Toggle token count display\n"
    "  [cyan]/cls[/cyan]                     Clear screen (keep context)\n\n"
    "[bold yellow]Agent Tools[/bold yellow]\n"
    "  [cyan]/shell[/cyan] [dim]<cmd>[/dim]             Run shell command, inject output\n"
    "  [cyan]/file[/cyan] [dim]<path>[/dim]             Load file into context\n"
    "  [cyan]/fetch[/cyan] [dim]<url>[/dim]             Fetch webpage into context\n"
    "  [cyan]/ls[/cyan] [dim]<path>[/dim]               List directory into context\n\n"
    "[bold yellow]Shortcuts[/bold yellow]\n"
    "  [cyan]Ctrl+C[/cyan]  Cancel/exit    [cyan]\u2191\u2193[/cyan]  Input history\n"
)

TOOL_SCHEMA = """
You are an autonomous agent. Call tools by responding with a JSON block like:

```tool
{
  "tool": "<tool_name>",
  "args": { ... }
}
```

Available tools:
- shell:  {"command": "bash command"}
- file:   {"path": "/path/to/file"}
- fetch:  {"url": "https://..."}
- ls:     {"path": "/some/dir"}
- done:   {}

Rules:
- Call ONE tool per response.
- After a tool result, continue reasoning then call the next tool or done.
- If you can answer without tools, just answer normally.
"""

PLAN_SCHEMA = """
You are a task planner. Break a goal into concrete steps.
Respond ONLY with a JSON object, no extra text:

{
  "goal": "<the goal>",
  "steps": [
    {"id": 1, "description": "...", "tool": "shell|file|fetch|ls|think|done", "args": "..."}
  ]
}
"""


class OllamaCLI:
    def __init__(self, model_override=None):
        self.config        = load_config()
        self.model         = model_override or self.config["model"]
        self.console       = Console()
        self.messages      = []
        self.system_prompt = None
        self.memories      = load_memory()
        self.auto_mode          = self.config.get("auto_mode", False)
        self.context_injections = []   # files/shells/fetches injected by user
        self.session            = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            style=Style.from_dict({"prompt": "bold ansicyan"}),
        )

    def print_startup(self):
        LOGO = [
            "  \u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557     \u2588\u2588\u2557      \u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2557   \u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2588\u2557 ",
            " \u2588\u2588\u2554\u2550\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551     \u2588\u2588\u2551     \u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557",
            " \u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2551     \u2588\u2588\u2551     \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551\u2588\u2588\u2554\u2588\u2588\u2588\u2588\u2554\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551",
            " \u2588\u2588\u2551   \u2588\u2588\u2551\u2588\u2588\u2551     \u2588\u2588\u2551     \u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2551\u2588\u2588\u2551\u255a\u2588\u2588\u2554\u255d\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2551",
            " \u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2551 \u255a\u2550\u255d \u2588\u2588\u2551\u2588\u2588\u2551  \u2588\u2588\u2551",
            "  \u255a\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u255d     \u255a\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u255d",
            "              C L I  v 3 . 0  \u00b7  A G E N T I C         ",
        ]
        colors = ["bright_cyan","cyan","bright_green","green","cyan","bright_cyan","dim white"]
        self.console.print()
        for line, color in zip(LOGO, colors):
            self.console.print(f"[{color}]{line}[/{color}]")
            time.sleep(0.04)
        self.console.print()

        tagline = "  Your autonomous local AI, beautifully in the terminal."
        rendered = ""
        for char in tagline:
            rendered += char
            self.console.print(f"[dim]{rendered}[/dim]", end="\r")
            time.sleep(0.015)
        self.console.print()
        self.console.print()

        ollama_ok = _is_ollama_running()
        try:
            import ollama as _ol
            raw         = _ol.list()
            entries     = raw.models if hasattr(raw, "models") else raw.get("models", [])
            model_count = len(entries)
        except Exception:
            model_count = 0

        checks = [
            (f"Ollama server    {'running' if ollama_ok else 'not running'}", ollama_ok),
            (f"Model            {self.model}", True),
            (f"Models available {model_count} installed", model_count > 0),
            (f"Memories loaded  {len(self.memories)} stored", True),
            (f"Auto mode        {'ON' if self.auto_mode else 'off'}", True),
        ]
        for label, ok in checks:
            icon = "[bold green]\u2713[/bold green]" if ok else "[bold red]\u2717[/bold red]"
            self.console.print(f"  {icon}  [dim]{label}[/dim]")
            time.sleep(0.08)

        self.console.print()
        self.console.rule(style=COLORS["rule"])
        self.console.print()

    def print_banner(self):
        auto_indicator = " [bold green]\u26a1 AUTO[/bold green]" if self.auto_mode else ""
        self.console.print(Panel(
            f"[bold green]Ollama CLI v3[/bold green]  \u00b7  model: [cyan]{self.model}[/cyan]{auto_indicator}\n"
            "[dim]/help for commands \u00b7 /exit to quit \u00b7 /auto to toggle agent mode[/dim]",
            border_style=COLORS["banner"], box=box.ROUNDED,
        ))
        self.console.print()

    def p_user(self, text):
        self.console.print(Panel(text,
            title=f"[bold {COLORS['user']}]You[/bold {COLORS['user']}]",
            border_style=COLORS["user"], box=box.ROUNDED))

    def p_ai(self, text, label=None):
        title = label or self.model
        self.console.print(Panel(Markdown(text),
            title=f"[bold {COLORS['ai']}]{title}[/bold {COLORS['ai']}]",
            border_style=COLORS["ai"], box=box.ROUNDED))

    def p_tool(self, title, body):
        self.console.print(Panel(f"[dim]{str(body)[:1500]}[/dim]",
            title=f"[bold {COLORS['tool']}]\u2699 {title}[/bold {COLORS['tool']}]",
            border_style=COLORS["tool"], box=box.ROUNDED))

    def p_step(self, n, total, desc, status="running"):
        icons  = {"running": "\u25b6", "done": "\u2713", "error": "\u2717"}
        colors = {"running": "yellow",  "done": "green",   "error": "red"}
        c = colors.get(status, "yellow")
        i = icons.get(status, "\u25b6")
        self.console.print(f"  [{c}]{i}[/{c}]  [dim]Step {n}/{total}:[/dim] {desc}")

    def err(self, msg):  self.console.print(f"[{COLORS['err']}]\u2717 {msg}[/{COLORS['err']}]")
    def info(self, msg): self.console.print(f"[{COLORS['info']}]\u2192 {msg}[/{COLORS['info']}]")
    def ok(self, msg):   self.console.print(f"[bold green]\u2713 {msg}[/bold green]")

    def _llm(self, messages, system=None):
        all_msgs = []
        if system:
            all_msgs.append({"role": "system", "content": system})
        all_msgs.extend(messages)
        try:
            resp = ollama.chat(model=self.model, messages=all_msgs)
            return resp["message"]["content"]
        except Exception as e:
            return f"[LLM error: {e}]"

    def _stream(self, messages, system=None):
        all_msgs = []
        if system:
            all_msgs.append({"role": "system", "content": system})
        all_msgs.extend(messages)
        full = ""
        try:
            stream = ollama.chat(model=self.model, messages=all_msgs, stream=True)
            self.console.print()
            self.console.print(f"[bold {COLORS['ai']}]\u258c {self.model}[/bold {COLORS['ai']}]")
            self.console.print(Rule(style=COLORS["rule"]))
            with Live(console=self.console, refresh_per_second=15) as live:
                for chunk in stream:
                    full += chunk["message"]["content"]
                    live.update(Markdown(full))
            self.console.print(Rule(style=COLORS["rule"]))
            if self.config.get("show_tokens"):
                self.console.print(f"[dim]~{int(len(full.split())*1.3)} tokens[/dim]")
            self.console.print()
        except Exception as e:
            self.err(f"Stream error: {e}")
        return full

    def _run_tool(self, tool, args):
        try:
            if tool == "shell":
                cmd = args if isinstance(args, str) else args.get("command", "")
                self.p_tool("shell", f"$ {cmd}")
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                return (r.stdout + r.stderr).strip() or "(no output)"
            elif tool == "file":
                path = args if isinstance(args, str) else args.get("path", "")
                p = Path(path).expanduser()
                if not p.exists():
                    return f"Error: file not found: {p}"
                content = p.read_text(errors="replace")
                self.p_tool(f"file: {p.name}", content[:300] + "...")
                return content
            elif tool == "fetch":
                if not WEB_ENABLED:
                    return "Error: pip install requests beautifulsoup4"
                url = args if isinstance(args, str) else args.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script","style","nav","footer"]):
                    tag.decompose()
                text = "\n".join(l for l in soup.get_text("\n", strip=True).splitlines() if l)[:4000]
                self.p_tool(f"fetch: {url[:50]}", text[:300] + "...")
                return text
            elif tool == "ls":
                path = args if isinstance(args, str) else args.get("path", ".")
                p = Path(path).expanduser()
                entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
                lines = [f"{'\U0001f4c1' if e.is_dir() else '\U0001f4c4'} {e.name}" for e in entries[:80]]
                out = "\n".join(lines)
                self.p_tool(f"ls: {p}", out[:300])
                return out
            elif tool == "done":
                return "__DONE__"
            else:
                return f"Error: unknown tool '{tool}'"
        except Exception as e:
            return f"Tool error: {e}"

    def _parse_tool_call(self, text):
        match = re.search(r"```tool\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(1))
            return data.get("tool"), data.get("args", {})
        except Exception:
            return None

    def _auto_chat(self, user_input):
        self.console.print(f"\n[bold yellow]⚡ AUTO MODE[/bold yellow] [dim]— agent running...[/dim]\n")
        mem_context = self._memory_context()
        base   = self._build_system()
        system = TOOL_SCHEMA + (f"\n\n{base}" if base else "")
        loop_msgs = list(self.messages) + [{"role": "user", "content": user_input}]
        max_steps = self.config.get("max_auto_steps", 10)
        final_response = ""
        for step in range(1, max_steps + 1):
            self.console.print(f"[dim]  step {step}/{max_steps}...[/dim]")
            response = self._llm(loop_msgs, system=system)
            tool_call = self._parse_tool_call(response)
            if tool_call is None:
                final_response = response
                break
            tool_name, tool_args = tool_call
            if tool_name == "done":
                final_response = re.sub(r"```tool.*?```", "", response, flags=re.DOTALL).strip()
                break
            tool_result = self._run_tool(tool_name, tool_args)
            loop_msgs.append({"role": "assistant", "content": response})
            loop_msgs.append({"role": "user", "content": f"[Tool result: {tool_name}]\n{tool_result}"})
        else:
            final_response = f"[Agent reached max steps ({max_steps})]"
        if final_response:
            self.p_ai(final_response)
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": final_response or "(agent used tools)"})

    def cmd_run(self, args):
        if not args:
            self.err("Usage: /run <file.py>")
            return
        filepath = Path(args.strip()).expanduser()
        if not filepath.exists():
            self.err(f"File not found: {filepath}")
            return
        max_iters = self.config.get("max_debug_iters", 5)
        self.console.print(f"\n[bold yellow]\U0001f501 Debug Loop[/bold yellow] \u2014 [cyan]{filepath.name}[/cyan] [dim](max {max_iters} iterations)[/dim]\n")
        current_code = filepath.read_text()
        for iteration in range(1, max_iters + 1):
            self.console.print(f"[bold]Iteration {iteration}/{max_iters}[/bold]")
            self.console.rule(style=COLORS["rule"])
            result = subprocess.run([sys.executable, str(filepath)], capture_output=True, text=True, timeout=30)
            if result.stdout.strip():
                self.p_tool("stdout", result.stdout.strip())
            if result.stderr.strip():
                self.p_tool("stderr", result.stderr.strip())
            if result.returncode == 0:
                self.ok(f"Code ran successfully on iteration {iteration}!")
                self.messages.append({"role": "user", "content": f"Ran `{filepath.name}` \u2014 success on iteration {iteration}.\nOutput:\n{result.stdout}"})
                return
            self.console.print(f"\n[yellow]\u26a0 Error \u2014 asking model to fix...[/yellow]\n")
            fix_prompt = (
                f"Fix this Python file. Return ONLY the corrected code in a ```python ... ``` block.\n\n"
                f"File: {filepath.name}\n\nCurrent code:\n```python\n{current_code}\n```\n\n"
                f"Error:\n```\n{result.stderr or result.stdout}\n```"
            )
            fix_response = self._llm([{"role": "user", "content": fix_prompt}])
            code_match = re.search(r"```python\s*(.*?)\s*```", fix_response, re.DOTALL)
            if not code_match:
                code_match = re.search(r"```\s*(.*?)\s*```", fix_response, re.DOTALL)
            if not code_match:
                self.err("Model didn't return a code block. Stopping.")
                self.p_ai(fix_response)
                return
            fixed_code = code_match.group(1)
            self.console.print(Panel(Markdown(f"```python\n{fixed_code}\n```"),
                title=f"[bold green]Model Fix \u2014 iteration {iteration}[/bold green]",
                border_style="green", box=box.ROUNDED))
            filepath.write_text(fixed_code)
            current_code = fixed_code
            self.console.print()
        self.err(f"Could not fix after {max_iters} iterations.")

    def cmd_plan(self, args):
        if not args:
            self.err("Usage: /plan <goal>")
            return
        goal = args.strip()
        self.console.print(f"\n[bold yellow]\U0001f4cb Planning:[/bold yellow] {goal}\n")
        raw = self._llm([{"role": "user", "content": f"Goal: {goal}"}], system=PLAN_SCHEMA)
        try:
            clean = re.sub(r"```json|```", "", raw).strip()
            plan  = json.loads(clean)
            steps = plan.get("steps", [])
        except Exception:
            self.err("Model didn't return a valid plan.")
            self.p_ai(raw)
            return
        if not steps:
            self.err("Plan has no steps.")
            return
        self.console.print(Panel(
            "\n".join(f"  [cyan]{s['id']}.[/cyan] {s['description']}  [dim]({s['tool']})[/dim]" for s in steps),
            title=f"[bold yellow]Plan: {goal}[/bold yellow]",
            border_style="yellow", box=box.ROUNDED))
        self.console.print()
        context = f"Goal: {goal}\n\n"
        for step in steps:
            sid, desc, tool, sargs = step["id"], step["description"], step.get("tool","think"), step.get("args","")
            total = len(steps)
            self.p_step(sid, total, desc, "running")
            if tool in ("think", "done"):
                result = self._llm([{"role": "user", "content": f"{context}\nNow complete step {sid}: {desc}\nBe concise."}])
                self.p_ai(result, label=f"Step {sid}")
                context += f"\nStep {sid}:\n{result}\n"
            else:
                result = self._run_tool(tool, sargs)
                context += f"\nStep {sid} ({tool}):\n{result}\n"
            self.p_step(sid, total, desc, "done")
            self.console.print()
        summary = self._llm([{"role": "user", "content": f"Summarize what was accomplished:\n{context}"}])
        self.console.print(Panel(Markdown(summary),
            title="[bold green]\u2705 Plan Complete \u2014 Summary[/bold green]",
            border_style="green", box=box.ROUNDED))
        self.messages.append({"role": "user",      "content": f"/plan {goal}"})
        self.messages.append({"role": "assistant",  "content": summary})

    def _memory_context(self):
        if not self.memories:
            return ""
        return "\n".join(f"- [{m['id']}] {m['content']}  ({m['date']})" for m in self.memories[-20:])

    def cmd_remember(self, args):
        if not args:
            self.err("Usage: /remember <fact>")
            return
        mem = {"id": len(self.memories) + 1, "content": args.strip(),
               "date": datetime.datetime.now().strftime("%Y-%m-%d")}
        self.memories.append(mem)
        save_memory(self.memories)
        self.ok(f"Stored memory [{mem['id']}]: {mem['content']}")

    def cmd_memories(self):
        if not self.memories:
            self.info("No memories yet. Use /remember <fact>")
            return
        table = Table(title="Long-Term Memories", box=box.ROUNDED, border_style="bright_black")
        table.add_column("ID",     style="cyan",  width=4)
        table.add_column("Memory", style="white")
        table.add_column("Date",   style="dim",   width=12)
        for m in self.memories:
            table.add_row(str(m["id"]), m["content"], m["date"])
        self.console.print(table)

    def cmd_forget(self, args):
        if not args.strip().isdigit():
            self.err("Usage: /forget <id>")
            return
        mid = int(args.strip())
        before = len(self.memories)
        self.memories = [m for m in self.memories if m["id"] != mid]
        if len(self.memories) < before:
            save_memory(self.memories)
            self.ok(f"Deleted memory {mid}")
        else:
            self.err(f"No memory with id {mid}")

    def cmd_auto(self):
        self.auto_mode = not self.auto_mode
        self.config["auto_mode"] = self.auto_mode
        save_config(self.config)
        if self.auto_mode:
            self.console.print("[bold green]\u26a1 Auto mode ON[/bold green] \u2014 model calls tools autonomously")
        else:
            self.console.print("[dim]Auto mode OFF \u2014 normal chat[/dim]")

    def tool_shell(self, cmd):
        out = self._run_tool("shell", cmd)
        self.context_injections.append(f"[Shell output: `{cmd}`]\n```\n{out}\n```")
        self.info(f"Shell output injected into context. ({len(self.context_injections)} active injections)")

    def tool_file(self, path):
        out = self._run_tool("file", path)
        self.context_injections.append(f"[File: `{path}`]\n```\n{out}\n```")
        self.console.print(f"[bold green]✓ File injected:[/bold green] [cyan]{path}[/cyan]  "
                           f"[dim]({len(out):,} chars · {len(self.context_injections)} injection(s) active)[/dim]")

    def tool_fetch(self, url):
        p = Path(url.strip()).expanduser()
        if p.exists():
            self.info("Looks like a local file — using /file instead.")
            self.tool_file(str(p))
            return
        out = self._run_tool("fetch", url)
        self.context_injections.append(f"[Webpage: {url}]\n{out}")
        self.info(f"Webpage injected into context. ({len(self.context_injections)} active injections)")

    def tool_ls(self, path="."):
        out = self._run_tool("ls", path)
        self.context_injections.append(f"[Directory listing: `{path}`]\n```\n{out}\n```")
        self.info(f"Directory listing injected into context. ({len(self.context_injections)} active injections)")

    def cmd_help(self):
        self.console.print(Panel(HELP_TEXT, border_style="yellow", box=box.ROUNDED))

    def cmd_models(self):
        try:
            raw     = ollama.list()
            # Newer ollama lib returns ListResponse with .models as objects
            # Older versions return a dict with "models" key
            entries = raw.models if hasattr(raw, "models") else raw.get("models", [])
            if not entries:
                self.info("No models. Run: ollama pull <model>")
                return
            table = Table(title="Local Ollama Models", box=box.ROUNDED, border_style="bright_black")
            table.add_column("Model",    style="cyan")
            table.add_column("Size",     style="green")
            table.add_column("Modified", style="dim")
            for m in entries:
                # Handle both object-style and dict-style entries
                if hasattr(m, "model"):
                    name    = m.model or m.name or "?"
                    size    = getattr(m, "size", 0) or 0
                    mod_raw = str(getattr(m, "modified_at", "") or "")
                else:
                    name    = m.get("name") or m.get("model") or "?"
                    size    = m.get("size", 0) or 0
                    mod_raw = str(m.get("modified_at", "") or "")
                size_str = f"{size / 1e9:.1f} GB"
                mod      = mod_raw[:10] if mod_raw else "?"
                marker   = " [bold yellow]\u2190 current[/bold yellow]" if name == self.model else ""
                table.add_row(name + marker, size_str, mod)
            self.console.print(table)
        except Exception as e:
            self.err(f"Could not list models: {e}")

    def cmd_model(self, args):
        if args:
            self.model = args.strip()
            self.config["model"] = self.model
            save_config(self.config)
            self.ok(f"Switched to: [cyan]{self.model}[/cyan]")
            return
        try:
            raw     = ollama.list()
            entries = raw.models if hasattr(raw, "models") else raw.get("models", [])
            models  = []
            for m in entries:
                if hasattr(m, "model"):
                    models.append(m.model or m.name or "?")
                else:
                    models.append(m.get("name") or m.get("model") or "?")
            models = [m for m in models if m and m != "?"]
        except Exception:
            models = []
        if not models:
            self.err("No local models found. Use /install to download one.")
            return
        # Arrow-key picker showing currently active model
        choices = [(m, f"{m}  ← current" if m == self.model else m) for m in models]
        selected = _arrow_select("Switch Model", choices, confirm_text="Select")
        if selected:
            self.model = selected
            self.config["model"] = self.model
            save_config(self.config)
            self.ok(f"Switched to: [cyan]{self.model}[/cyan]")
        else:
            self.info("Cancelled.")

    def cmd_current(self):
        """Show the currently active model."""
        self.console.print(Panel(
            f"[bold green]{self.model}[/bold green]",
            title="[bold cyan]Active Model[/bold cyan]",
            border_style="cyan", box=box.ROUNDED
        ))

    def cmd_install(self):
        """Browse and install any model from the full catalogue."""
        selected = _arrow_select(
            "Install a Model",
            INSTALLABLE_MODELS,
            confirm_text="Install",
        )
        if not selected:
            self.info("Cancelled.")
            return
        # Check if already installed
        try:
            raw     = ollama.list()
            entries = raw.models if hasattr(raw, "models") else raw.get("models", [])
            installed = set()
            for m in entries:
                installed.add(m.model if hasattr(m, "model") else m.get("name",""))
        except Exception:
            installed = set()

        if selected in installed:
            self.console.print(f"[yellow]→ {selected} is already installed.[/yellow]")
            switch = self.console.input("[bold]Set it as active model? (y/N):[/bold] ").strip().lower()
            if switch == "y":
                self.model = selected
                self.config["model"] = selected
                save_config(self.config)
                self.ok(f"Active model set to: [cyan]{selected}[/cyan]")
            return

        ok = _do_pull(self.console, selected, self.config, set_as_default=False)
        if ok:
            switch = self.console.input("[bold]Set as active model? (y/N):[/bold] ").strip().lower()
            if switch == "y":
                self.model = selected
                self.config["model"] = selected
                save_config(self.config)
                self.ok(f"Active model set to: [cyan]{selected}[/cyan]")

    def cmd_compare(self, args):
        parts = args.strip().split()
        try:
            raw         = ollama.list()
            entries     = raw.models if hasattr(raw, "models") else raw.get("models", [])
            models_list = []
            for m in entries:
                if hasattr(m, "model"):
                    models_list.append(m.model or m.name or "?")
                else:
                    models_list.append(m.get("name") or m.get("model") or "?")
            models_list = [m for m in models_list if m and m != "?"]
        except Exception:
            models_list = []
        if len(parts) >= 2:
            m1, m2 = parts[0], parts[1]
        elif len(models_list) >= 2:
            self.console.print("[bold yellow]Pick two models:[/bold yellow]")
            for i, m in enumerate(models_list, 1):
                self.console.print(f"  [cyan]{i}.[/cyan] {m}")
            try:
                c1 = self.console.input("[bold]Model 1:[/bold] ").strip()
                c2 = self.console.input("[bold]Model 2:[/bold] ").strip()
                m1 = models_list[int(c1) - 1] if c1.isdigit() else c1
                m2 = models_list[int(c2) - 1] if c2.isdigit() else c2
            except (KeyboardInterrupt, EOFError, IndexError):
                self.info("Cancelled.")
                return
        else:
            self.err("Need at least 2 local models.")
            return
        self.console.print(f"\n[bold]Comparing [cyan]{m1}[/cyan] vs [cyan]{m2}[/cyan][/bold]")
        try:
            user_input = self.session.prompt(HTML("<ansiyellow>prompt \u203a </ansiyellow>")).strip()
        except (KeyboardInterrupt, EOFError):
            return
        if not user_input:
            return
        self.p_user(user_input)
        responses = {}
        saved_model = self.model
        for mdl in [m1, m2]:
            self.model = mdl
            self.console.print(f"\n[bold {COLORS['ai']}]\u258c {mdl}[/bold {COLORS['ai']}]")
            self.console.print(Rule(style=COLORS["rule"]))
            full = ""
            try:
                stream = ollama.chat(model=mdl, messages=[{"role": "user", "content": user_input}], stream=True)
                with Live(console=self.console, refresh_per_second=15) as live:
                    for chunk in stream:
                        full += chunk["message"]["content"]
                        live.update(Markdown(full))
            except Exception as e:
                full = f"[Error: {e}]"
                self.console.print(f"[red]{full}[/red]")
            responses[mdl] = full
            self.console.print(Rule(style=COLORS["rule"]))
        self.model = saved_model
        table = Table(box=box.ROUNDED, border_style="bright_black")
        table.add_column(m1, style="cyan",  ratio=1)
        table.add_column(m2, style="green", ratio=1)
        table.add_row(responses.get(m1,"")[:300] + "...", responses.get(m2,"")[:300] + "...")
        table.add_row(f"[dim]{len(responses.get(m1,'').split())} words[/dim]",
                      f"[dim]{len(responses.get(m2,'').split())} words[/dim]")
        self.console.print("\n[bold yellow]\u2500\u2500 Side-by-Side \u2500\u2500[/bold yellow]")
        self.console.print(table)

    def _cls(self):
        """Clear the terminal screen without affecting conversation context."""
        os.system("clear" if os.name != "nt" else "cls")

    def cmd_clear(self):
        self.messages = []
        self.system_prompt = None
        self._cls()
        self.print_banner()
        self.ok("Conversation cleared")

    def cmd_retry(self):
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()
        last = next((m["content"] for m in reversed(self.messages) if m["role"] == "user"), None)
        if not last:
            self.err("No previous message to retry.")
            return
        self.info("Retrying...")
        resp = self._stream(self.messages, system=self._build_system())
        if resp:
            self.messages.append({"role": "assistant", "content": resp})

    def cmd_system(self, args):
        if not args:
            if self.system_prompt:
                self.console.print(Panel(self.system_prompt, title="System Prompt", border_style="yellow"))
            else:
                self.info("No system prompt set. Usage: /system <prompt>")
            return
        self.system_prompt = args.strip()
        self.ok("System prompt updated.")

    def cmd_persona(self, name):
        if not name:
            self.err("Usage: /persona <n>")
            return
        path = PERSONAS_DIR / f"{name}.txt"
        if not path.exists():
            self.err(f"Persona '{name}' not found. Use /personas to list.")
            return
        self.system_prompt = path.read_text().strip()
        self.ok(f"Loaded persona: [cyan]{name}[/cyan]")

    def cmd_personas(self):
        files = sorted(PERSONAS_DIR.glob("*.txt"))
        if not files:
            self.info("No personas saved yet.")
            return
        table = Table(title="Personas", box=box.ROUNDED, border_style="bright_black")
        table.add_column("Name",    style="cyan")
        table.add_column("Preview", style="dim")
        for f in files:
            content = f.read_text().strip()
            table.add_row(f.stem, content[:70] + ("..." if len(content) > 70 else ""))
        self.console.print(table)

    def cmd_save_persona(self, name):
        if not name:
            self.err("Usage: /save-persona <n>")
            return
        if not self.system_prompt:
            self.err("No system prompt set.")
            return
        (PERSONAS_DIR / f"{name}.txt").write_text(self.system_prompt)
        self.ok(f"Persona saved: [cyan]{name}[/cyan]")

    def cmd_save(self, args):
        name = args.strip() or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not name.endswith(".json"):
            name += ".json"
        (SAVES_DIR / name).write_text(json.dumps({
            "model": self.model, "system_prompt": self.system_prompt,
            "messages": self.messages,
            "saved_at": datetime.datetime.now().isoformat(),
        }, indent=2))
        self.ok(f"Saved: [cyan]{name}[/cyan]")

    def cmd_load(self, args):
        name = args.strip()
        if not name:
            self.err("Usage: /load <n>")
            return
        if not name.endswith(".json"):
            name += ".json"
        path = SAVES_DIR / name
        if not path.exists():
            self.err(f"Not found: {path}")
            return
        d = json.loads(path.read_text())
        self.model         = d.get("model", self.model)
        self.system_prompt = d.get("system_prompt")
        self.messages      = d.get("messages", [])
        self.ok(f"Loaded: [cyan]{name}[/cyan] ({len(self.messages)} messages)")

    def cmd_list(self):
        files = sorted(SAVES_DIR.glob("*.json"))
        if not files:
            self.info("No saved conversations.")
            return
        table = Table(title="Saved Conversations", box=box.ROUNDED, border_style="bright_black")
        table.add_column("Name",     style="cyan")
        table.add_column("Model",    style="green")
        table.add_column("Messages", style="yellow")
        table.add_column("Saved At", style="dim")
        for f in files:
            try:
                d = json.loads(f.read_text())
                table.add_row(f.stem, d.get("model","?"),
                    str(len(d.get("messages",[]))), d.get("saved_at","?")[:19])
            except Exception:
                table.add_row(f.stem, "?", "?", "?")
        self.console.print(table)

    def cmd_context(self):
        """Show or clear active context injections."""
        if not self.context_injections:
            self.info("No context injections active.")
            return
        self.console.print(f"[bold yellow]{len(self.context_injections)} active injection(s):[/bold yellow]")
        for i, inj in enumerate(self.context_injections, 1):
            preview = inj.splitlines()[0][:80]
            self.console.print(f"  [cyan]{i}.[/cyan] [dim]{preview}...[/dim]")
        self.console.print()
        ans = self.console.input("[bold]Clear all injections? (y/N):[/bold] ").strip().lower()
        if ans == "y":
            self.context_injections = []
            self.ok("Context injections cleared.")

    def cmd_info(self):
        table = Table(box=box.ROUNDED, border_style="bright_black", show_header=False)
        table.add_column("Key",   style="bold yellow")
        table.add_column("Value", style="cyan")
        table.add_row("Model",     self.model)
        table.add_row("Auto mode", "ON \u26a1" if self.auto_mode else "off")
        table.add_row("Messages",  str(len(self.messages)))
        table.add_row("Memories",  str(len(self.memories)))
        table.add_row("System",    (self.system_prompt or "none")[:60])
        table.add_row("Config",    str(CONFIG_PATH))
        self.console.print(table)

    def cmd_tokens(self):
        self.config["show_tokens"] = not self.config.get("show_tokens", False)
        save_config(self.config)
        self.ok(f"Token display: {'ON' if self.config['show_tokens'] else 'OFF'}")

    def _build_system(self):
        """Assemble the full system prompt from memories + injections + user system prompt."""
        parts = []
        mem = self._memory_context()
        if mem:
            parts.append(f"Memories about the user:\n{mem}")
        if self.context_injections:
            joined = "\n\n".join(self.context_injections)
            parts.append(
                "IMPORTANT: The user has loaded the following content into your context.\n"
                "You MUST use this content to answer questions. Do NOT say you cannot see files or code.\n"
                "Treat this content as if you wrote it yourself and know it completely.\n\n"
                + joined
            )
        if self.system_prompt:
            parts.append(self.system_prompt)
        return "\n\n".join(parts) or None

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        resp = self._stream(self.messages, system=self._build_system())
        if resp:
            self.messages.append({"role": "assistant", "content": resp})

    def run(self):
        self.print_startup()
        self.print_banner()
        while True:
            try:
                label      = "\u26a1 you" if self.auto_mode else "you"
                user_input = self.session.prompt(HTML(f"<b>{label}</b> <ansiyellow>\u203a</ansiyellow> ")).strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Goodbye![/dim]")
                break
            if not user_input:
                continue
            # Ctrl+L equivalent
            if user_input in ("\x0c",):
                self._cls()
                continue
            if user_input.startswith("/"):
                parts = user_input[1:].split(" ", 1)
                cmd   = parts[0].lower()
                args  = parts[1] if len(parts) > 1 else ""
                match cmd:
                    case "help":         self.cmd_help()
                    case "models":       self.cmd_models()
                    case "model":        self.cmd_model(args)
                    case "current":      self.cmd_current()
                    case "install":      self.cmd_install()
                    case "compare":      self.cmd_compare(args)
                    case "clear":        self.cmd_clear()
                    case "cls":          self._cls()
                    case "retry":        self.cmd_retry()
                    case "system":       self.cmd_system(args)
                    case "persona":      self.cmd_persona(args)
                    case "personas":     self.cmd_personas()
                    case "save-persona": self.cmd_save_persona(args)
                    case "save":         self.cmd_save(args)
                    case "load":         self.cmd_load(args)
                    case "list":         self.cmd_list()
                    case "info":         self.cmd_info()
                    case "context":      self.cmd_context()
                    case "tokens":       self.cmd_tokens()
                    case "shell":        self.tool_shell(args)
                    case "file":         self.tool_file(args)
                    case "fetch":        self.tool_fetch(args)
                    case "ls":           self.tool_ls(args or ".")
                    case "auto":         self.cmd_auto()
                    case "plan":         self.cmd_plan(args)
                    case "run":          self.cmd_run(args)
                    case "remember":     self.cmd_remember(args)
                    case "memories":     self.cmd_memories()
                    case "forget":       self.cmd_forget(args)
                    case "exit" | "quit":
                        self.console.print("[dim]Goodbye![/dim]")
                        return
                    case _:
                        self.err(f"Unknown command: /{cmd}  (try /help)")
            else:
                if self.auto_mode:
                    self._auto_chat(user_input)
                else:
                    self.chat(user_input)


def main():
    parser = argparse.ArgumentParser(description="Ollama CLI v3 — Agentic")
    parser.add_argument("--model",         "-m",  help="Model to use")
    parser.add_argument("--compare",       action="store_true", help="Start in compare mode")
    parser.add_argument("--auto",          action="store_true", help="Start with auto mode on")
    parser.add_argument("--no-bootstrap",  action="store_true", help="Skip Ollama setup checks")
    args = parser.parse_args()

    console = Console()
    config  = load_config()

    # Run bootstrap unless explicitly skipped
    if not args.no_bootstrap:
        ok = bootstrap(console, config)
        if not ok:
            sys.exit(1)

    app = OllamaCLI(model_override=args.model)
    if args.auto:
        app.auto_mode = True
    if args.compare:
        app.print_startup()
        app.print_banner()
        app.cmd_compare("")
    else:
        app.run()


if __name__ == "__main__":
    main()