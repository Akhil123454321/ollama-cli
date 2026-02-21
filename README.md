# Ollama CLI — Terminal UI for Local LLMs

A beautiful, agentic CLI for running local Ollama models with tool-calling, memory, planning, and compare modes.

Quick links
- [src/main.py](src/main.py) — main v3 implementation (contains [`OllamaCLI`](src/main.py) and `main()` entry)
- [ollama_cli.py](ollama_cli.py) — simpler v1 CLI
- [ollama_cli_v2.py](ollama_cli_v2.py) — v2 CLI with themes and expanded features
- [pyproject.toml](pyproject.toml) — project metadata & dependencies

Requirements
- Python >= 3.10
- Install runtime deps listed in [pyproject.toml](pyproject.toml):
  pip install rich prompt_toolkit ollama requests beautifulsoup4

Run
- From the package entry (if installed): ollama-cli (configured in [pyproject.toml](pyproject.toml): `ollama_cli.main:main`)
- Directly:
  - v3: python src/main.py
  - v2: python ollama_cli_v2.py
  - v1: python ollama_cli.py

Features
- Interactive chat with streaming output and Markdown rendering
- Agentic mode: auto tool-calling (shell, file, fetch, ls) and iterative plan execution
- /plan: generate and execute step plans
- /run: debug loop for running and auto-fixing Python scripts
- Long-term memories (/remember, /memories, /forget)
- Model management and side-by-side model comparison
- Save/load conversations and personas

Key symbols
- [`OllamaCLI`](src/main.py) — primary CLI class and command implementations
- [`bootstrap`](src/main.py) — ensures Ollama is installed/running and can pull models
- [`_run_tool`](src/main.py) — executes shell/file/fetch/ls tool calls

Config & data
- Config stored at: `~/.ollama_cli_config.json`
- History: `~/.ollama_cli_history`
- Saves: `~/.ollama_cli_saves`
- Personas: `~/.ollama_cli_personas`
- Memory: `~/.ollama_cli_memory.json`

Development
- Run unit tests (if any) via pytest (dev deps in pyproject)
- Linting and formatting recommended with tools of choice

Notes
- The CLI requires the `ollama` binary or library; v3 bootstraps installation attempts via `bootstrap` in [src/main.py](src/main.py).
- Web fetch features require `requests` and `beautifulsoup4` (optional).

License
- MIT — see LICENSE

Contributing
- PRs and issues welcome. Keep changes focused and include tests where appropriate.