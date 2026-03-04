#!/usr/bin/env python3
"""
Memabra Skill - 反馈学习脚本

用法:
    # 提供一次完整交互的反馈
    python3 feedback.py --user-input "用户问题" --response "助理回复" --followup "用户后续回复"

    # 手动指定奖励值
    python3 feedback.py --user-input "用户问题" --strategy "direct_answer" --reward 0.8
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

DATA_DIR = os.path.expanduser("~/.memabra")
MODEL_PATH = os.path.join(DATA_DIR, "model.pt")
MEMORY_PATH = os.path.join(DATA_DIR, "memories.json")
STATS_PATH = os.path.join(DATA_DIR, "stats.json")


def get_agent():
    """Get MemabraAgent with persistence."""
    from memabra.agent import MemabraAgent

    os.makedirs(DATA_DIR, exist_ok=True)

    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    agent = MemabraAgent(model_path=model_path)

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

    stats = agent.get_stats()
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2, default=str)


def feedback_from_followup(user_input: str, response: str, followup: str) -> dict:
    """Evaluate feedback from a complete interaction cycle."""
    agent = get_agent()

    # Simulate the interaction
    result = agent.process(user_input)

    # Override the response in conversation history
    if agent.conversation_history:
        agent.conversation_history[-1]['content'] = response

    # Evaluate followup
    feedback = agent.on_user_followup(followup)
    save_agent(agent)

    return {
        "strategy_used": result["strategy"],
        "confidence": result["confidence"],
        "feedback": feedback,
    }


def feedback_manual(user_input: str, strategy: str, reward: float) -> dict:
    """Manually provide reward feedback."""
    agent = get_agent()

    query_emb = agent.embedder(user_input)
    update_stats = agent.update_from_feedback(
        query_embedding=query_emb,
        strategy_id=strategy,
        reward=reward
    )
    save_agent(agent)

    return {
        "strategy": strategy,
        "reward": reward,
        "update": update_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Memabra Feedback Learning")
    parser.add_argument("--user-input", required=True, help="Original user input")
    parser.add_argument("--response", type=str, help="Assistant's response")
    parser.add_argument("--followup", type=str, help="User's followup message")
    parser.add_argument("--strategy", type=str, help="Strategy used (for manual feedback)")
    parser.add_argument("--reward", type=float, help="Manual reward value (-1.0 to 1.0)")
    args = parser.parse_args()

    if args.followup and args.response:
        result = feedback_from_followup(args.user_input, args.response, args.followup)
    elif args.strategy is not None and args.reward is not None:
        result = feedback_manual(args.user_input, args.strategy, args.reward)
    else:
        print(json.dumps({
            "error": "Provide either (--response + --followup) or (--strategy + --reward)"
        }))
        sys.exit(1)

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
