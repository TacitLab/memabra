# Memabra

**Memabra** = **Mem**ory + Cere**bra** (Brain Cortex)

A bio-inspired memory and intuition system for AI agents.

🌏 [中文版本 (Chinese Version)](README_CN.md)

---

## Core Philosophy

When humans encounter a problem, they first rely on **intuition** to quickly judge how to approach it, then retrieve relevant memories, and finally execute. If the result is effective, this path is reinforced; if it fails, deep search and pattern correction are triggered.

Memabra brings this dual-system thinking (intuition + analysis) into AI agent memory management.

## Design Principles

1. **Bio-inspired**: Drawing from human memory formation, consolidation, and forgetting mechanisms
2. **Implicit Feedback**: No explicit user ratings needed—signals are read naturally from interaction flow
3. **End-to-End Learning**: The intuition network self-optimizes through experience
4. **Progressive Complexity**: Start with simple prototypes, evolve gradually

## Project Structure

```
memabra/
├── src/memabra/
│   ├── __init__.py               # Package exports
│   ├── agent.py                  # MemabraAgent — core integration
│   ├── intuition_network.py      # PyTorch intuition network (dual-head: strategy + memory query)
│   ├── intuition.py              # Lightweight intuition network (cosine-similarity based)
│   ├── feedback_evaluator.py     # Implicit feedback evaluator + delayed reward assigner + calibrator
│   ├── feedback.py               # Simplified feedback system
│   └── memory.py                 # Hierarchical memory (episodic / semantic / procedural)
├── skill/                        # OpenClaw Skill integration
│   ├── SKILL.md                  # Skill description & SOP
│   └── scripts/
│       ├── ensure_env.sh         # Environment setup script
│       ├── predict.py            # Intuition prediction CLI
│       ├── memorize.py           # Memory management CLI
│       └── feedback.py           # Feedback learning CLI
├── tests/                        # Test suite (60 test cases)
│   ├── test_intuition.py         # Intuition network tests
│   ├── test_memory.py            # Memory system tests
│   ├── test_feedback.py          # Feedback system tests
│   ├── test_agent.py             # Agent integration tests
│   └── test_skill_scripts.py     # Skill CLI end-to-end tests
├── config/
│   └── default.yaml              # Default configuration
├── docs/                         # Design documents
└── pyproject.toml                # Project build config
```

## Key Modules

### IntuitionNetwork (`intuition_network.py`)
A PyTorch neural policy network with dual heads:
- **Strategy Head**: Selects the best action strategy (direct answer, search, tool use, clarification)
- **Memory Query Head**: Generates a query vector for memory retrieval

Trained via REINFORCE policy gradient with entropy regularization and adaptive temperature.

### HierarchicalMemory (`memory.py`)
Three-layer memory system inspired by human cognition:
- **Episodic Memory**: Records of specific interactions and outcomes
- **Semantic Memory**: Factual knowledge as subject-predicate-object triples
- **Procedural Memory**: Skills and procedures with trigger patterns

Supports Ebbinghaus forgetting curve and disk persistence (save/load).

### ImplicitEvaluator (`feedback_evaluator.py`)
Infers user satisfaction without explicit ratings by detecting:
- Positive/negative keywords
- Give-up signals
- Semantic repetition (user repeating the same question)
- Rephrasing patterns
- Topic shifts with contextual satisfaction inference

### MemabraAgent (`agent.py`)
Orchestrates the full loop: embed input → predict strategy → retrieve memories → execute → collect feedback → update network.

## Quick Start

```bash
# Install in development mode
pip install -e .

# Run the built-in demo
python -m memabra.agent
```

### Usage Example

```python
from memabra import MemabraAgent

agent = MemabraAgent()

# Process user input
result = agent.process("What's the weather today?")
print(result['response'])
print(f"Strategy: {result['strategy']}, Confidence: {result['confidence']:.2f}")

# Provide follow-up (feedback is inferred automatically)
feedback = agent.on_user_followup("Thanks, got it!")
print(f"Feedback: {feedback['feedback_type']}, Reward: {feedback['reward']:+.2f}")

# Save model state
agent.save("model.pt")
```

## OpenClaw Skill Integration

Memabra can be used as an [OpenClaw](https://github.com/jasons20/openclaw) Skill, providing AI agents with bio-inspired memory and intuition capabilities via CLI scripts.

### CLI Scripts

```bash
# Intuition prediction — get strategy recommendation for user input
python3 skill/scripts/predict.py --query "How to optimize database performance"

# Memory store — save episodic/semantic/procedural memories
python3 skill/scripts/memorize.py --action store --type episodic --content "User asked about SQL optimization"

# Memory search — retrieve relevant memories
python3 skill/scripts/memorize.py --action search --query "user preferences" --top-k 5

# Feedback learning — update intuition network from user followup
python3 skill/scripts/feedback.py --user-input "Help me optimize SQL" --response "You can add indexes" --followup "Great, 10x faster!"

# System stats
python3 skill/scripts/predict.py --stats
```

See [`skill/SKILL.md`](skill/SKILL.md) for the full SOP with 5 usage scenarios.

## Testing

60 test cases covering all modules:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v
```

| Test File | Coverage | Cases |
|-----------|----------|-------|
| `test_intuition.py` | Intuition network | 12 |
| `test_memory.py` | Hierarchical memory | 12 |
| `test_feedback.py` | Feedback evaluator | 12 |
| `test_agent.py` | Agent integration | 10 |
| `test_skill_scripts.py` | Skill CLI (end-to-end) | 14 |

## Documentation

- [Architecture](docs/architecture.md) - System architecture overview
- [Core Mechanisms](docs/core-mechanisms.md) - Intuition network, memory retrieval, execution loop
- [Feedback System](docs/feedback-system.md) - Inferring satisfaction from user behavior
- [Roadmap](docs/roadmap.md) - Implementation plan

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0

## License

MIT
