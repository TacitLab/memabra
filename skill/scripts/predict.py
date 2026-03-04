#!/usr/bin/env python3
"""
Memabra Skill - 直觉预测脚本

用法:
    python3 predict.py --query "用户输入"
    python3 predict.py --stats
"""

import argparse
import json
import os
import sys

# Ensure memabra is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

DATA_DIR = os.path.expanduser("~/.memabra")
MODEL_PATH = os.path.join(DATA_DIR, "model.pt")
MEMORY_PATH = os.path.join(DATA_DIR, "memories.json")
STATS_PATH = os.path.join(DATA_DIR, "stats.json")


def get_agent():
    """Get or create MemabraAgent with persistence."""
    from memabra.agent import MemabraAgent

    os.makedirs(DATA_DIR, exist_ok=True)

    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    agent = MemabraAgent(model_path=model_path)

    # Load memories if available
    if os.path.exists(MEMORY_PATH):
        try:
            agent.memory.load_from_disk(MEMORY_PATH)
        except Exception:
            pass

    return agent


def save_agent(agent):
    """Persist agent state."""
    agent.save(MODEL_PATH)
    agent.memory.save_to_disk(MEMORY_PATH)

    # Save stats
    stats = agent.get_stats()
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2, default=str)


def predict(query: str) -> dict:
    """Run intuition prediction on a query."""
    agent = get_agent()
    result = agent.process(query)
    save_agent(agent)
    return result


def get_stats() -> dict:
    """Get current system statistics."""
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, 'r') as f:
            return json.load(f)

    agent = get_agent()
    return agent.get_stats()


def main():
    parser = argparse.ArgumentParser(description="Memabra Intuition Prediction")
    parser.add_argument("--query", type=str, help="User input to predict strategy for")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    args = parser.parse_args()

    if args.stats:
        result = get_stats()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif args.query:
        result = predict(args.query)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
