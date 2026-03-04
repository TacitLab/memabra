#!/usr/bin/env python3
"""
Memabra Skill - 记忆管理脚本

用法:
    # 存储情景记忆
    python3 memorize.py --action store --type episodic --content "交互内容"

    # 存储语义记忆
    python3 memorize.py --action store --type semantic --subject "主语" --predicate "谓语" --object "宾语"

    # 存储程序记忆
    python3 memorize.py --action store --type procedural --name "技能名" --trigger "触发词1,触发词2" --action-desc "执行动作"

    # 搜索记忆
    python3 memorize.py --action search --query "搜索内容" --top-k 5

    # 列出最近记忆
    python3 memorize.py --action recent --top-k 10
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

DATA_DIR = os.path.expanduser("~/.memabra")
MEMORY_PATH = os.path.join(DATA_DIR, "memories.json")


def get_memory():
    """Get HierarchicalMemory with persistence."""
    from memabra.agent import DummyEmbedder
    from memabra.memory import HierarchicalMemory

    os.makedirs(DATA_DIR, exist_ok=True)

    embedder = DummyEmbedder()
    memory = HierarchicalMemory(embedding_fn=embedder)

    if os.path.exists(MEMORY_PATH):
        try:
            memory.load_from_disk(MEMORY_PATH)
        except Exception:
            pass

    return memory


def save_memory(memory):
    """Save memory to disk."""
    memory.save_to_disk(MEMORY_PATH)


def store_episodic(memory, content: str) -> dict:
    """Store an episodic memory."""
    mem_id = memory.episodic.add_interaction(
        input_text=content,
        output_text="",
        strategy="manual_store"
    )
    save_memory(memory)
    return {"status": "ok", "type": "episodic", "id": mem_id}


def store_semantic(memory, subject: str, predicate: str, obj: str) -> dict:
    """Store a semantic memory."""
    mem_id = memory.semantic.add_fact(
        subject=subject,
        predicate=predicate,
        obj=obj,
        source="skill_input",
        confidence=0.8
    )
    save_memory(memory)
    return {"status": "ok", "type": "semantic", "id": mem_id}


def store_procedural(memory, name: str, triggers: str, action_desc: str) -> dict:
    """Store a procedural memory."""
    trigger_list = [t.strip() for t in triggers.split(",")]
    mem_id = memory.procedural.add_skill(
        name=name,
        trigger_patterns=trigger_list,
        action=action_desc
    )
    save_memory(memory)
    return {"status": "ok", "type": "procedural", "id": mem_id}


def search_memories(memory, query: str, top_k: int = 5) -> dict:
    """Search across all memory types."""
    results = memory.retrieve(
        query_text=query,
        strategy_id="direct_answer",
        top_k=top_k
    )

    output = {"query": query, "results": {}}
    for mem_type, mems in results.items():
        output["results"][mem_type] = [
            {
                "id": m.id,
                "content": m.content,
                "strength": round(m.strength, 3),
                "access_count": m.access_count,
            }
            for m in mems
        ]

    total = sum(len(v) for v in output["results"].values())
    output["total_found"] = total
    return output


def recent_memories(memory, top_k: int = 10) -> dict:
    """Get recent episodic memories."""
    recent = memory.episodic.get_recent_context(top_k)
    return {
        "recent": [
            {
                "id": m.id,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
                "strength": round(m.strength, 3),
            }
            for m in recent
        ],
        "count": len(recent),
    }


def main():
    parser = argparse.ArgumentParser(description="Memabra Memory Management")
    parser.add_argument("--action", required=True, choices=["store", "search", "recent"])
    parser.add_argument("--type", choices=["episodic", "semantic", "procedural"], default="episodic")
    parser.add_argument("--content", type=str, help="Content for episodic memory")
    parser.add_argument("--subject", type=str, help="Subject for semantic memory")
    parser.add_argument("--predicate", type=str, help="Predicate for semantic memory")
    parser.add_argument("--object", type=str, help="Object for semantic memory")
    parser.add_argument("--name", type=str, help="Skill name for procedural memory")
    parser.add_argument("--trigger", type=str, help="Comma-separated triggers for procedural memory")
    parser.add_argument("--action-desc", type=str, help="Action description for procedural memory")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    memory = get_memory()

    if args.action == "store":
        if args.type == "episodic":
            if not args.content:
                print(json.dumps({"error": "--content is required for episodic memory"}))
                sys.exit(1)
            result = store_episodic(memory, args.content)
        elif args.type == "semantic":
            if not all([args.subject, args.predicate, args.object]):
                print(json.dumps({"error": "--subject, --predicate, --object are required"}))
                sys.exit(1)
            result = store_semantic(memory, args.subject, args.predicate, args.object)
        elif args.type == "procedural":
            if not all([args.name, args.trigger, args.action_desc]):
                print(json.dumps({"error": "--name, --trigger, --action-desc are required"}))
                sys.exit(1)
            result = store_procedural(memory, args.name, args.trigger, args.action_desc)
        else:
            result = {"error": f"Unknown type: {args.type}"}

    elif args.action == "search":
        if not args.query:
            print(json.dumps({"error": "--query is required for search"}))
            sys.exit(1)
        result = search_memories(memory, args.query, args.top_k)

    elif args.action == "recent":
        result = recent_memories(memory, args.top_k)

    else:
        result = {"error": f"Unknown action: {args.action}"}

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
