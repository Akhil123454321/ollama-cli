#!/usr/bin/env python3
"""
ollama_cli.py — A rich CLI interface for Ollama
Install deps: pip install rich prompt_toolkit ollama
Run: python ollama_cli.py
"""

import json
import os
import sys
import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.rule import Rule
    from rich import box
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.formatted_text import HTML
    import ollama
except ImportError:
    print("Missing dependencies. Run: pip install rich prompt_toolkit ollama")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "llama3.1:8b"
HISTORY_FILE = Path.home() / ".ollama_cli_history"
SAVES_DIR = Path.home() / ".ollama_cli_saves"
SAVES_DIR.mkdir(exist_ok=True)

THEME = {
    "user_color": "bold cyan",
    "assistant_color": "bold green",
    "system_color": "bold yellow",
    "error_color": "bold red",
    "info_color": "dim white",
    "border_color": "bright_black",
}

PROMPT_STYLE = Style.from_dict({
    "prompt": "cyan bold",
})

HELP_TEXT = """
[bold yellow]Available Commands[/bold yellow]

  [cyan]/help[/cyan]              Show this help message
  [cyan]/model[/cyan]             Switch model interactively
  [cyan]/model <name>[/cyan]      Switch to a specific model (e.g. /model mistral:7b)
  [cyan]/models[/cyan]            List all locally available Ollama models
  [cyan]/clear[/cyan]             Clear conversation history
  [cyan]/save [name][/cyan]       Save conversation to file (default: timestamp)
  [cyan]/load <name>[/cyan]       Load a saved conversation
  [cyan]/list[/cyan]              List saved conversations
  [cyan]/system <msg>[/cyan]      Set a system prompt
  [cyan]/info[/cyan]              Show current session info
  [cyan]/exit[/cyan]  [cyan]/quit[/cyan]     Exit the CLI

[dim]Tip: Your input history is saved across sessions.[/dim]
"""


# ── App State ─────────────────────────────────────────────────────────────────

class OllamaCLI:
    def __init__(self):
        self.console = Console()
        self.model = DEFAULT_MODEL
        self.messages: list[dict] = []
        self.system_prompt: str | None = None
        self.session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            style=PROMPT_STYLE,
        )

    # ── UI Helpers ─────────────────────────────────────────────────────────

    def print_banner(self):
        self.console.print()
        self.console.print(Panel(
            f"[bold green]Ollama CLI[/bold green]  ·  model: [cyan]{self.model}[/cyan]\n"
            "[dim]Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit[/dim]",
            border_style="bright_black",
            box=box.ROUNDED,
        ))
        self.console.print()

    def print_user(self, text: str):
        self.console.print(Panel(
            text,
            title=f"[{THEME['user_color']}]You[/{THEME['user_color']}]",
            border_style="cyan",
            box=box.ROUNDED,
        ))

    def print_assistant(self, text: str):
        self.console.print(Panel(
            Markdown(text),
            title=f"[{THEME['assistant_color']}]{self.model}[/{THEME['assistant_color']}]",
            border_style="green",
            box=box.ROUNDED,
        ))

    def print_error(self, text: str):
        self.console.print(f"[{THEME['error_color']}]✗ {text}[/{THEME['error_color']}]")

    def print_info(self, text: str):
        self.console.print(f"[{THEME['info_color']}]→ {text}[/{THEME['info_color']}]")

    def print_success(self, text: str):
        self.console.print(f"[bold green]✓ {text}[/bold green]")

    # ── Commands ───────────────────────────────────────────────────────────

    def cmd_help(self):
        self.console.print(Panel(HELP_TEXT, border_style="yellow", box=box.ROUNDED))

    def cmd_models(self):
        try:
            result = ollama.list()
            models = result.get("models", [])
            if not models:
                self.print_info("No models found. Run: ollama pull <model>")
                return

            table = Table(title="Local Ollama Models", box=box.ROUNDED, border_style="bright_black")
            table.add_column("Model", style="cyan")
            table.add_column("Size", style="green")
            table.add_column("Modified", style="dim")

            for m in models:
                name = m.get("name", "unknown")
                size_bytes = m.get("size", 0)
                size_gb = f"{size_bytes / 1e9:.1f} GB" if size_bytes else "?"
                modified = m.get("modified_at", "")[:10] if m.get("modified_at") else "?"
                marker = " [bold yellow]← current[/bold yellow]" if name == self.model else ""
                table.add_row(name + marker, size_gb, modified)

            self.console.print(table)
        except Exception as e:
            self.print_error(f"Could not list models: {e}")

    def cmd_model(self, args: str):
        if args:
            self.model = args.strip()
            self.print_success(f"Switched to model: [cyan]{self.model}[/cyan]")
        else:
            # Interactive picker
            try:
                result = ollama.list()
                models = [m["name"] for m in result.get("models", [])]
            except Exception:
                models = []

            if not models:
                self.print_error("No local models found. Pull one with: ollama pull <model>")
                return

            self.console.print("\n[bold yellow]Available models:[/bold yellow]")
            for i, m in enumerate(models, 1):
                marker = " [bold cyan]← current[/bold cyan]" if m == self.model else ""
                self.console.print(f"  [cyan]{i}.[/cyan] {m}{marker}")

            try:
                choice = self.console.input("\n[bold]Enter number or model name:[/bold] ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(models):
                        self.model = models[idx]
                    else:
                        self.print_error("Invalid selection")
                        return
                else:
                    self.model = choice
                self.print_success(f"Switched to model: [cyan]{self.model}[/cyan]")
            except (KeyboardInterrupt, EOFError):
                self.print_info("Model switch cancelled")

    def cmd_clear(self):
        self.messages = []
        self.system_prompt = None
        self.console.clear()
        self.print_banner()
        self.print_success("Conversation cleared")

    def cmd_system(self, args: str):
        if not args:
            if self.system_prompt:
                self.console.print(Panel(self.system_prompt, title="Current System Prompt", border_style="yellow"))
            else:
                self.print_info("No system prompt set. Usage: /system <your prompt>")
            return
        self.system_prompt = args.strip()
        self.print_success(f"System prompt set: [dim]{self.system_prompt[:60]}...[/dim]" if len(self.system_prompt) > 60 else f"System prompt set.")

    def cmd_save(self, args: str):
        name = args.strip() if args else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not name.endswith(".json"):
            name += ".json"
        path = SAVES_DIR / name
        data = {
            "model": self.model,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "saved_at": datetime.datetime.now().isoformat(),
        }
        path.write_text(json.dumps(data, indent=2))
        self.print_success(f"Saved to [cyan]{path}[/cyan]")

    def cmd_load(self, args: str):
        name = args.strip()
        if not name:
            self.print_error("Usage: /load <name>")
            return
        if not name.endswith(".json"):
            name += ".json"
        path = SAVES_DIR / name
        if not path.exists():
            self.print_error(f"File not found: {path}")
            return
        data = json.loads(path.read_text())
        self.model = data.get("model", self.model)
        self.system_prompt = data.get("system_prompt")
        self.messages = data.get("messages", [])
        self.print_success(f"Loaded [cyan]{name}[/cyan] ({len(self.messages)} messages, model: {self.model})")

    def cmd_list(self):
        files = sorted(SAVES_DIR.glob("*.json"))
        if not files:
            self.print_info(f"No saved conversations in {SAVES_DIR}")
            return
        table = Table(title="Saved Conversations", box=box.ROUNDED, border_style="bright_black")
        table.add_column("Name", style="cyan")
        table.add_column("Saved At", style="dim")
        table.add_column("Messages", style="green")
        for f in files:
            try:
                data = json.loads(f.read_text())
                saved_at = data.get("saved_at", "?")[:19]
                msg_count = str(len(data.get("messages", [])))
            except Exception:
                saved_at, msg_count = "?", "?"
            table.add_row(f.stem, saved_at, msg_count)
        self.console.print(table)

    def cmd_info(self):
        table = Table(box=box.ROUNDED, border_style="bright_black", show_header=False)
        table.add_column("Key", style="bold yellow")
        table.add_column("Value", style="cyan")
        table.add_row("Model", self.model)
        table.add_row("Messages", str(len(self.messages)))
        table.add_row("System prompt", self.system_prompt or "[dim]none[/dim]")
        table.add_row("History file", str(HISTORY_FILE))
        table.add_row("Saves dir", str(SAVES_DIR))
        self.console.print(table)

    # ── Chat ───────────────────────────────────────────────────────────────

    def chat(self, user_input: str):
        self.print_user(user_input)
        self.messages.append({"role": "user", "content": user_input})

        all_messages = []
        if self.system_prompt:
            all_messages.append({"role": "system", "content": self.system_prompt})
        all_messages.extend(self.messages)

        # Stream response with live display
        full_response = ""
        try:
            stream = ollama.chat(model=self.model, messages=all_messages, stream=True)

            self.console.print()
            self.console.print(f"[{THEME['assistant_color']}]▌ {self.model}[/{THEME['assistant_color']}]")
            self.console.print(Rule(style="bright_black"))

            with Live(console=self.console, refresh_per_second=15) as live:
                for chunk in stream:
                    delta = chunk["message"]["content"]
                    full_response += delta
                    live.update(Markdown(full_response))

            self.console.print(Rule(style="bright_black"))
            self.console.print()

        except ollama.ResponseError as e:
            self.print_error(f"Ollama error: {e}")
            self.messages.pop()  # remove the user message we just added
            return
        except Exception as e:
            self.print_error(f"Unexpected error: {e}")
            self.messages.pop()
            return

        self.messages.append({"role": "assistant", "content": full_response})

    # ── Main Loop ──────────────────────────────────────────────────────────

    def run(self):
        self.print_banner()

        while True:
            try:
                prompt_text = HTML(f'<cyan><b>you</b></cyan> <ansiyellow>›</ansiyellow> ')
                user_input = self.session.prompt(prompt_text, style=PROMPT_STYLE).strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            # Command routing
            if user_input.startswith("/"):
                parts = user_input[1:].split(" ", 1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                match cmd:
                    case "help":       self.cmd_help()
                    case "models":     self.cmd_models()
                    case "model":      self.cmd_model(args)
                    case "clear":      self.cmd_clear()
                    case "system":     self.cmd_system(args)
                    case "save":       self.cmd_save(args)
                    case "load":       self.cmd_load(args)
                    case "list":       self.cmd_list()
                    case "info":       self.cmd_info()
                    case "exit" | "quit": 
                        self.console.print("[dim]Goodbye![/dim]")
                        break
                    case _:
                        self.print_error(f"Unknown command: /{cmd}  (type /help for list)")
            else:
                self.chat(user_input)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Optional: pass model as CLI arg
    if len(sys.argv) > 1:
        app = OllamaCLI()
        app.model = sys.argv[1]
    else:
        app = OllamaCLI()
    app.run()