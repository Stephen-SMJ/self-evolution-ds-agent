# KAIROS — Memory System

> This feature exists in the official AutoDS codebase but has not been fully released. autods implements and ships it.

The assistant can remember information across sessions and automatically consolidate memories over time.

## Commands

| Command | Description |
|---------|-------------|
| `/remember <text>` | Save a note to the daily log |
| `/memory` | Show current memory index |
| `/dream` | Manually consolidate daily logs into organized topic files |

## Auto-Dream

Runs automatically after a turn when:
- >= 24 hours since last consolidation
- >= 5 new sessions since last consolidation

Configurable: `--dream-interval`, `--dream-min-sessions`, `--no-auto-dream`

## Try It Out

```bash
autods --auto-approve
> /remember I prefer Python over JavaScript
> /remember Our project uses gRPC + PostgreSQL
> /dream                    # consolidate into topic files
> /memory                   # verify the index

# New session — the model recalls your preferences
autods
> What do you know about my preferences?
```

Data stored in `~/.autods/` (memory in `memory/`, sessions in `sessions/`).
