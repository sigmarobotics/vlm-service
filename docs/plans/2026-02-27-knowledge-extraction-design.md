# Knowledge Extraction & Memory Consolidation Design

**Date:** 2026-02-27
**Status:** Approved

## Goal

Scan all session logs and project codebases to:
1. Extract lessons learned, pitfalls, and effective patterns into auto memory
2. Identify shared code patterns across projects for extraction as skills/plugins/libraries

## Scope

- **Projects:** All ~15 projects in `~/CodeBase/`
- **Logs:** `~/.claude/history.jsonl` + all per-project session logs under `~/.claude/projects/`
- **Output:** Topic-based memory files + skill/plugin candidate list

## Agent Architecture

### Phase 1 — Parallel Scanning (concurrent)

| Agent | Mission | Input |
|-------|---------|-------|
| **Session Log Miner** | Extract error patterns, debugging breakthroughs, workflow friction, repeated mistakes, solutions | `~/.claude/history.jsonl`, `~/.claude/projects/*/` session logs |
| **Kachaka Code Scanner** | Identify duplicated code, shared patterns, abstraction opportunities | `sdk-toolkit`, `bio-patrol`, `visual-patrol`, `connection-test`, `kachaka-api`, `kachaka-api-TW`, `kachaka-cmd-center`, `kachaka-gemini` |
| **Non-Kachaka Code Scanner** | Same as above for non-Kachaka projects | `LiveAPI-integration`, `claude-cowork-UI-test`, `test-claude-skills`, `vila-test`, `visual-patrol-v2`, `Sigma-patrol` |

### Phase 2 — Synthesis (after Phase 1)

| Agent | Mission | Input |
|-------|---------|-------|
| **Memory Writer** | Organize findings into topic-based memory files | Reports from Phase 1 agents |
| **Skill/Plugin Designer** | Propose extractions with type, rationale, and priority | Shared-patterns findings from code scanners |

## Memory Output Structure

```
~/.claude/projects/-home-snaken--claude/memory/
  MEMORY.md                  # Updated index (< 200 lines)
  kachaka.md                 # Enriched (already exists)
  hooks-config.md            # Preserved (already exists)
  kachaka-patterns.md        # NEW: Common Kachaka code patterns
  debugging-lessons.md       # NEW: Debugging techniques
  common-pitfalls.md         # NEW: Mistakes and how to avoid them
  shared-utilities.md        # NEW: Code to extract/share
  workflow-tips.md           # NEW: Effective Claude Code workflows
  skill-plugin-candidates.md # NEW: Proposed skills/plugins to build
```

Each file: concise bullet points with context and source references, < 150 lines.

## Extraction Criteria

| Type | When to use | Examples |
|------|-------------|---------|
| **Skill** | Repeatable workflow/approach pattern | "How to debug Kachaka connections", "How to set up a patrol project" |
| **Plugin** | Reusable code with Claude-specific context | Camera utilities, connection helpers, common MCP patterns |
| **Shared library** | Pure utility code, no Claude context | Data transforms, protocol helpers |

Output: prioritized candidate list with project references, extraction type, and estimated complexity.

## Key Decisions

- Topic-based memory organization (not per-project) for cross-cutting discoverability
- 3 parallel scan agents + 2 sequential synthesis agents
- Kachaka and non-Kachaka scanning split for focused analysis
- Existing memory files preserved and enriched, not overwritten
