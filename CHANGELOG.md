# Changelog

All notable changes to ollama-cli will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2026-02-21

### 🎉 Initial Release

#### Core Chat
- Streaming markdown responses with live rendering
- Syntax-highlighted code blocks
- Persistent input history across sessions (`↑↓` to navigate)
- Conversation save/load as JSON (`/save`, `/load`, `/list`)
- System prompt management (`/system`)
- Retry last response (`/retry`)

#### Model Management
- Arrow-key model picker (`/model`)
- Full model catalogue with `/install` — browse 25+ models, install with Enter
- `/current` — show active model at a glance
- `/models` — list all locally installed models
- Side-by-side model comparison (`/compare`)

#### Agentic Features
- **Auto mode** (`/auto`) — model autonomously calls tools to complete tasks
- **Iterative debug loop** (`/run file.py`) — auto-fixes errors until code passes
- **Plan executor** (`/plan <goal>`) — breaks goals into steps and executes each
- **Long-term memory** (`/remember`) — facts persist across sessions

#### Agent Tools
- `/shell <cmd>` — run shell commands, inject output into context
- `/file <path>` — load file contents into context
- `/fetch <url>` — fetch webpage text into context
- `/ls <path>` — inject directory listing into context
- `/context` — view or clear active injections

#### Quality of Life
- Auto-detects and installs Ollama if missing (macOS/Linux)
- Auto-starts `ollama serve` if not running
- First-run model picker on fresh install
- `/cls` and `Ctrl+L` to clear screen without losing context
- `/clear` to reset conversation
- Personas system (`/persona`, `/personas`, `/save-persona`)
- Token count display (`/tokens`)

---

## Upcoming

- [ ] MCP server — expose tools to Claude Code, Cursor, and other agents
- [ ] Repo-aware context — auto-index codebase on launch
- [ ] Git tools — `/diff`, `/commit`, `/log`
- [ ] API key integrations — Claude, OpenAI, Gemini, Groq
- [ ] Symbol search across codebase
- [ ] Web search tool in auto mode