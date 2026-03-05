# Memabra

**Memabra** = **Mem**ory + Cere**bra** (Brain Cortex)

A bio-inspired memory and intuition **Skill** for AI agents, designed for [OpenClaw](https://github.com/jasons20/openclaw).

🌏 [中文版本 (Chinese Version)](README_CN.md)

---

## What It Does

When humans encounter a problem, they first rely on **intuition** to quickly judge how to approach it, then retrieve relevant memories, and finally execute. If the result is effective, this path is reinforced; if it fails, deep search and pattern correction are triggered.

Memabra brings this dual-system thinking into AI agents as an OpenClaw Skill:

1. **Intuition Prediction** — Neural network selects the best strategy (direct answer / search / tool use / clarification)
2. **Hierarchical Memory** — Episodic, semantic, and procedural memory with forgetting curves
3. **Feedback Learning** — Implicit satisfaction inference from user behavior, online network updates

## Project Structure

```
memabra/
├── skill/                            # OpenClaw Skill (self-contained)
│   ├── SKILL.md                      # Skill description & SOP
│   ├── memabra/                      # Core runtime modules
│   │   ├── __init__.py               # Package exports
│   │   ├── agent.py                  # MemabraAgent — full agent
│   │   ├── intuition_network.py      # PyTorch dual-head network (strategy + memory query)
│   │   ├── feedback_evaluator.py     # Implicit feedback evaluator + delayed reward + calibrator
│   │   └── memory.py                 # Hierarchical memory (episodic / semantic / procedural)
│   └── scripts/                      # CLI entry points
│       ├── ensure_env.sh             # Environment setup
│       ├── predict.py                # Intuition prediction
│       ├── memorize.py               # Memory management
│       └── feedback.py               # Feedback learning
├── tests/                            # Test suite
│   ├── conftest.py                   # Path setup
│   ├── test_intuition.py             # Intuition network tests
│   ├── test_memory.py                # Memory system tests
│   ├── test_feedback.py              # Feedback system tests
│   ├── test_agent.py                 # Agent integration tests
│   └── test_skill_scripts.py         # Skill CLI end-to-end tests
├── config/
│   └── default.yaml                  # Default configuration
├── docs/                             # Design documents
└── pyproject.toml                    # Project build config
```

## Quick Start (OpenClaw Skill)

### Setup

```bash
cd skill
bash scripts/ensure_env.sh
```

### Usage

```bash
# Intuition prediction — get strategy for user input
python3 scripts/predict.py --query "How to optimize database performance"

# Store episodic memory
python3 scripts/memorize.py --action store --type episodic --content "User asked about SQL optimization"

# Store semantic memory (fact)
python3 scripts/memorize.py --action store --type semantic --subject "User" --predicate "prefers" --object "concise answers"

# Store procedural memory (skill)
python3 scripts/memorize.py --action store --type procedural --name "SQL Optimization" --trigger "optimize,index,slow query" --action-desc "Analyze slow query log and suggest indexes"

# Search memories
python3 scripts/memorize.py --action search --query "user preferences" --top-k 5

# Feedback learning from followup
python3 scripts/feedback.py --user-input "Help me optimize SQL" --response "You can add indexes" --followup "Great, 10x faster!"

# System stats
python3 scripts/predict.py --stats
```

See [`skill/SKILL.md`](skill/SKILL.md) for the full SOP with 5 usage scenarios.

## Core Modules

### IntuitionNetwork (`intuition_network.py`)
PyTorch neural policy network with dual heads:
- **Strategy Head**: Selects the best action strategy via temperature-scaled softmax
- **Memory Query Head**: Generates L2-normalized query vector for memory retrieval

Trained via REINFORCE policy gradient with entropy regularization.

### HierarchicalMemory (`memory.py`)
Three-layer memory system:
- **Episodic**: Records of specific interactions and outcomes
- **Semantic**: Factual knowledge as subject-predicate-object triples
- **Procedural**: Skills with trigger patterns and success tracking

Supports Ebbinghaus forgetting curve and disk persistence.

### ImplicitEvaluator (`feedback_evaluator.py`)
Infers user satisfaction without explicit ratings:
- Positive/negative keyword detection
- Give-up signal recognition
- Semantic repetition analysis
- Rephrasing pattern detection
- Contextual topic-shift satisfaction inference

### MemabraAgent (`agent.py`)
Full loop: embed input → predict strategy → retrieve memories → execute → collect feedback → update network.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Data Persistence

All data stored in `~/.memabra/`:
- `model.pt` — Intuition network weights
- `memories.json` — Memory data
- `stats.json` — Statistics

## Documentation

- [Architecture](docs/architecture.md) — System architecture overview
- [Core Mechanisms](docs/core-mechanisms.md) — Intuition network, memory retrieval, execution loop
- [Feedback System](docs/feedback-system.md) — Inferring satisfaction from user behavior
- [Roadmap](docs/roadmap.md) — Implementation plan

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.24.0

## License

MIT
