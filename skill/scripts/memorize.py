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

    # 存储动作链记忆（广义记忆：记录 tool/skill 使用经验）
    python3 memorize.py --action store --type action --query "用户的问题" --strategy "tool_use" --chain '[{"step_index":0,"action_type":"tool_call","tool_or_skill":"search_file","params":{"pattern":"*.py"},"result_summary":"found 5 files","success":true,"latency_ms":120}]' --response-summary "找到5个Python文件" --reward 0.8 --tags "file_search,python"

    # 搜索记忆
    python3 memorize.py --action search --query "搜索内容" --top-k 5

    # 列出最近记忆
    python3 memorize.py --action recent --top-k 10

    # 查看工具使用统计
    python3 memorize.py --action tool-stats

    # 查找某工具的成功案例
    python3 memorize.py --action tool-patterns --tool-name "search_file" --min-reward 0.5
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


def store_action(memory, query: str, strategy: str, chain_json: str,
                 response_summary: str, reward: float = 0.0,
                 tags: str = "", success: bool = True) -> dict:
    """Store an action chain memory (broad-sense memory)."""
    chain = json.loads(chain_json)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    
    mem_id = memory.action.record_action_chain(
        user_query=query,
        strategy_used=strategy,
        action_chain=chain,
        final_response_summary=response_summary,
        reward=reward,
        context_tags=tag_list,
        success=success,
    )
    save_memory(memory)
    return {"status": "ok", "type": "action", "id": mem_id}


def get_tool_stats(memory) -> dict:
    """Get aggregated tool usage statistics."""
    stats = memory.action.get_tool_stats()
    return {
        "tool_stats": {
            tool: {
                "total_calls": s["total_calls"],
                "success_rate": round(s["success_rate"], 3),
                "avg_latency_ms": round(s["avg_latency_ms"], 1),
                "avg_reward": round(s["avg_reward"], 3),
            }
            for tool, s in stats.items()
        },
        "total_tools": len(stats),
    }


def get_tool_patterns(memory, tool_name: str, min_reward: float = 0.5) -> dict:
    """Find successful usage patterns for a specific tool."""
    patterns = memory.action.find_successful_patterns(tool_name, min_reward)
    return {
        "tool": tool_name,
        "min_reward": min_reward,
        "patterns": [
            {
                "id": m.id,
                "user_query": m.user_query,
                "tools_used": m.tools_used,
                "total_steps": m.total_steps,
                "reward": round(m.reward, 3),
                "action_chain": m.action_chain,
            }
            for m in patterns
        ],
        "count": len(patterns),
    }


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
    parser.add_argument("--action", required=True,
                        choices=["store", "search", "recent", "tool-stats", "tool-patterns"])
    parser.add_argument("--type", choices=["episodic", "semantic", "procedural", "action"],
                        default="episodic")
    parser.add_argument("--content", type=str, help="Content for episodic memory")
    parser.add_argument("--subject", type=str, help="Subject for semantic memory")
    parser.add_argument("--predicate", type=str, help="Predicate for semantic memory")
    parser.add_argument("--object", type=str, help="Object for semantic memory")
    parser.add_argument("--name", type=str, help="Skill name for procedural memory")
    parser.add_argument("--trigger", type=str, help="Comma-separated triggers for procedural memory")
    parser.add_argument("--action-desc", type=str, help="Action description for procedural memory")
    # Action memory arguments
    parser.add_argument("--query", type=str, help="User query (for action memory or search)")
    parser.add_argument("--strategy", type=str, help="Strategy used (for action memory)")
    parser.add_argument("--chain", type=str, help="Action chain JSON (for action memory)")
    parser.add_argument("--response-summary", type=str, help="Response summary (for action memory)")
    parser.add_argument("--reward", type=float, default=0.0, help="Reward value (for action memory)")
    parser.add_argument("--tags", type=str, default="", help="Comma-separated context tags")
    parser.add_argument("--success", type=str, default="true", help="Whether chain succeeded (true/false)")
    # Tool analysis arguments
    parser.add_argument("--tool-name", type=str, help="Tool name for pattern analysis")
    parser.add_argument("--min-reward", type=float, default=0.5, help="Min reward for pattern filtering")
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
        elif args.type == "action":
            if not all([args.query, args.strategy, args.chain, args.response_summary]):
                print(json.dumps({"error": "--query, --strategy, --chain, --response-summary are required for action memory"}))
                sys.exit(1)
            success = args.success.lower() in ("true", "1", "yes")
            result = store_action(
                memory, args.query, args.strategy, args.chain,
                args.response_summary, args.reward, args.tags, success
            )
        else:
            result = {"error": f"Unknown type: {args.type}"}

    elif args.action == "search":
        if not args.query:
            print(json.dumps({"error": "--query is required for search"}))
            sys.exit(1)
        result = search_memories(memory, args.query, args.top_k)

    elif args.action == "recent":
        result = recent_memories(memory, args.top_k)

    elif args.action == "tool-stats":
        result = get_tool_stats(memory)

    elif args.action == "tool-patterns":
        if not args.tool_name:
            print(json.dumps({"error": "--tool-name is required for tool-patterns"}))
            sys.exit(1)
        result = get_tool_patterns(memory, args.tool_name, args.min_reward)

    else:
        result = {"error": f"Unknown action: {args.action}"}

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
