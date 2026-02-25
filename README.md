# ollama-agentic

A beautiful, agentic terminal interface for [Ollama](https://ollama.com) — run local LLMs with auto tool-calling, long-term memory, git integration, concurrent subagents, and semantic code search.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/pypi/v/ollama-agentic)

---

## ⚠️ Requirement: Ollama must be installed first

**This CLI is a frontend for Ollama. It will not work without Ollama installed and running on your machine.**

1. Download and install Ollama from [ollama.com/download](https://ollama.com/download)
2. Start it: `ollama serve` (or open the Ollama desktop app)
3. Pull a model: `ollama pull mistral` or `ollama pull llama3.1:8b`

Then install and launch this CLI:

```bash
pip install ollama-agentic
ollama-cli
```

---

## Features

- ⚡ **Auto mode** — model autonomously calls tools to complete tasks (`/auto`)
- 🐝 **Swarm agents** — `/swarm` splits complex tasks across parallel background agents
- 🔍 **Semantic code search (RAG)** — AST-aware local codebase indexing, no API needed
- 🌿 **Git integration** — `/git` status, diff, log, commit (with AI messages), branch, stash
- 🔁 **Iterative debug loop** — `/run file.py` auto-fixes errors until code passes
- 📋 **Plan executor** — `/plan <goal>` breaks goals into typed steps and executes them
- 🧠 **Long-term memory** — `/remember` stores facts that persist across sessions
- ⬇️ **Arrow-key model picker** — `/install` lets you browse and download 25+ models
- 🔧 **Agent tools** — `/shell`, `/file`, `/fetch`, `/ls` inject real context into chats
- 💾 **Conversation saving** — `/save` and `/load` persist chats as JSON
- 🎭 **Personas** — save and load system prompt presets
- 🆚 **Compare mode** — run the same prompt through two models side by side

---

## Usage

```bash
ollama-cli                       # start chatting
ollama-cli --model qwen2.5:7b    # start with a specific model
ollama-cli --auto                # start in autonomous agent mode
ollama-cli --compare             # compare two models side by side
```

---

## Commands

### Chat & Navigation
| Command | Description |
|---|---|
| `/cls` | Clear screen (keep context) |
| `/clear` | Clear conversation and screen |
| `Ctrl+L` | Clear screen |
| `/retry` | Regenerate last response |
| `/tokens` | Toggle token count display |

### Models
| Command | Description |
|---|---|
| `/model` | Switch active model (arrow-key picker) |
| `/current` | Show currently active model |
| `/install` | Browse & install models from catalogue |
| `/models` | List all installed models |
| `/compare` | Compare two models side by side |

### Agentic
| Command | Description |
|---|---|
| `/auto` | Toggle autonomous tool-calling mode |
| `/plan <goal>` | Break a goal into steps and execute |
| `/run <file.py>` | Run code, auto-fix errors in a loop |
| `/swarm <task>` | Decompose task across parallel background agents |
| `/swarm-status` | Check swarm progress |
| `/swarm-status full` | See full output from each agent |

### Git
| Command | Description |
|---|---|
| `/git` | Show git status |
| `/git diff` | Show unstaged diff, inject into context |
| `/git diff staged` | Show staged diff |
| `/git log` | Recent commits with timestamps |
| `/git branch` | List branches |
| `/git branch <n>` | Switch branch |
| `/git commit` | Stage and commit (AI message option) |
| `/git stash` | Stash changes |

### RAG — Semantic Code Search
| Command | Description |
|---|---|
| `/rag` | Show index status |
| `/rag index` | Incremental index of project |
| `/rag index full` | Wipe and rebuild index |
| `/rag search <query>` | Semantic search over codebase |
| `/rag auto` | Toggle auto-inject relevant chunks into every chat |
| `/rag clear` | Wipe the index |

### Memory
| Command | Description |
|---|---|
| `/remember <fact>` | Store a fact in long-term memory |
| `/memories` | List all stored memories |
| `/forget <id>` | Delete a memory by ID |

### Context Injection
| Command | Description |
|---|---|
| `/file <path>` | Load a file into context |
| `/shell <cmd>` | Run a shell command, inject output |
| `/fetch <url>` | Fetch a webpage into context |
| `/ls <path>` | Inject a directory listing |
| `/context` | View or clear active injections |

### Conversations & Personas
| Command | Description |
|---|---|
| `/save <n>` | Save conversation |
| `/load <n>` | Load conversation |
| `/list` | List saved conversations |
| `/system <prompt>` | Set a system prompt |
| `/persona <n>` | Load a saved persona |
| `/personas` | List saved personas |
| `/save-persona <n>` | Save current system prompt as persona |

---

## Swarm Agents

`/swarm` decomposes a complex task into independent subtasks and runs them as parallel agents in the background. You keep using the CLI while they work.

```
you › /swarm research React Server Components vs traditional SSR
you › /swarm-status          # check mid-task
you › /swarm-status full     # read each agent's full output
```

---

## RAG — Semantic Code Search

Run from inside any git repo. Uses AST-aware chunking for Python and sliding-window chunking for all other languages. Embeddings run fully offline via `sentence-transformers`.

RAG dependencies are optional — the CLI works fine without them:

```bash
pip install lancedb sentence-transformers tree-sitter tree-sitter-python
```

```
you › /rag index             # index your project (~seconds)
you › /rag search auth flow  # semantic search
you › /rag auto              # auto-inject relevant chunks into every chat
```

The index lives in `.ollama_rag/` inside your project. Only changed files are re-indexed on subsequent runs.

---

## Agent Mode

Toggle with `/auto` or launch with `--auto`. The model calls tools, reads results, and loops until the task is done.

```
⚡ you › look at main.py and find any bugs
⚡ you › write a web scraper for hacker news and run it
⚡ you › set up a basic Flask app in this folder
```

---

## Config & Data

| Path | Description |
|---|---|
| `~/.ollama_cli_config.json` | Settings (model, auto mode, etc) |
| `~/.ollama_cli_history` | Input history |
| `~/.ollama_cli_memory.json` | Long-term memories |
| `~/.ollama_cli_saves/` | Saved conversations |
| `~/.ollama_cli_personas/` | Saved personas |
| `.ollama_rag/` | RAG vector index (per project, inside project root) |

---

## Requirements

- Python 3.10+
- macOS, Linux, or Windows
- **Ollama installed and running** — [ollama.com/download](https://ollama.com/download)

---

## Roadmap

- [ ] Project memory — `/understand` deep-reads your codebase and stores structured knowledge
- [ ] MCP server support — connect to filesystem, GitHub, Postgres, browser tools
- [ ] TUI dashboard — split-pane interface with live swarm agent view
- [ ] API key integrations — Claude, OpenAI, Gemini, Groq as model backends

---

## Contributing

PRs and issues welcome at [github.com/Akhil123454321/ollama-cli](https://github.com/Akhil123454321/ollama-cli). Keep changes focused and include tests where appropriate.

## License

MIT — see [LICENSE](LICENSE)