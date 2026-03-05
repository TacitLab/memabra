"""Tests for Hierarchical Memory System."""

import numpy as np
import pytest
from datetime import datetime, timedelta


def make_embedder(dim=64):
    """Create a simple deterministic embedder for testing."""
    def embedder(text):
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(dim)
        return (vec / np.linalg.norm(vec)).tolist()
    return embedder


class TestMemory:
    """Tests for base Memory dataclass."""

    def test_create_memory(self):
        from memabra.memory import Memory
        m = Memory(content="test")
        assert m.content == "test"
        assert m.strength == 1.0
        assert m.access_count == 0
        assert m.id is not None


class TestMemoryStore:
    """Tests for MemoryStore base class."""

    def test_add_and_get(self):
        from memabra.memory import MemoryStore, Memory

        store = MemoryStore()
        m = Memory(content="hello")
        mid = store.add(m)

        retrieved = store.get(mid)
        assert retrieved is not None
        assert retrieved.content == "hello"
        assert retrieved.access_count == 1  # get increments access_count

    def test_get_nonexistent(self):
        from memabra.memory import MemoryStore
        store = MemoryStore()
        assert store.get("nonexistent") is None

    def test_search_by_embedding(self):
        from memabra.memory import MemoryStore, Memory

        embed = make_embedder()
        store = MemoryStore(embedding_fn=embed)

        store.add(Memory(content="Python programming", embedding=embed("Python programming")))
        store.add(Memory(content="Java development", embedding=embed("Java development")))
        store.add(Memory(content="cooking recipes", embedding=embed("cooking recipes")))

        query_emb = embed("Python coding")
        results = store.search(query_emb, top_k=2)
        assert len(results) <= 2

    def test_forgetting_curve(self):
        from memabra.memory import MemoryStore, Memory

        store = MemoryStore()
        m = Memory(content="old memory", strength=1.0)
        store.add(m)

        # Apply forgetting with time far in the future
        future = datetime.utcnow() + timedelta(days=30)
        store.apply_forgetting(future)

        assert m.strength < 1.0, "Memory strength should decay over time"


class TestEpisodicStore:
    """Tests for EpisodicStore."""

    def test_add_interaction(self):
        from memabra.memory import EpisodicStore

        store = EpisodicStore(embedding_fn=make_embedder())
        mid = store.add_interaction(
            input_text="how to sort a list",
            output_text="use sorted() function",
            strategy="direct_answer"
        )
        assert mid is not None

    def test_get_recent_context(self):
        from memabra.memory import EpisodicStore

        store = EpisodicStore(embedding_fn=make_embedder())
        for i in range(5):
            store.add_interaction(f"question {i}", f"answer {i}", "direct_answer")

        recent = store.get_recent_context(n=3)
        assert len(recent) == 3


class TestSemanticStore:
    """Tests for SemanticStore."""

    def test_add_fact(self):
        from memabra.memory import SemanticStore

        store = SemanticStore(embedding_fn=make_embedder())
        mid = store.add_fact(
            subject="Python",
            predicate="is",
            obj="a programming language",
            confidence=0.95
        )
        assert mid is not None


class TestProceduralStore:
    """Tests for ProceduralStore."""

    def test_add_and_find_skill(self):
        from memabra.memory import ProceduralStore

        store = ProceduralStore()
        store.add_skill(
            name="SQL optimization",
            trigger_patterns=["optimize", "slow query", "index"],
            action="Analyze and add indexes"
        )

        matches = store.find_matching_skills("how to optimize my database")
        assert len(matches) == 1
        assert matches[0].name == "SQL optimization"

    def test_no_matching_skill(self):
        from memabra.memory import ProceduralStore

        store = ProceduralStore()
        store.add_skill(name="test", trigger_patterns=["xyz"], action="do something")

        matches = store.find_matching_skills("unrelated query")
        assert len(matches) == 0


class TestHierarchicalMemory:
    """Tests for HierarchicalMemory."""

    def test_retrieve_by_strategy(self):
        from memabra.memory import HierarchicalMemory

        embed = make_embedder()
        hm = HierarchicalMemory(embedding_fn=embed)

        # Add some memories
        hm.episodic.add_interaction("test input", "test output", "direct_answer")
        hm.semantic.add_fact("AI", "is", "artificial intelligence")
        hm.procedural.add_skill("greeting", ["hello", "hi"], "Say hello back")

        # Test different strategies
        for strategy in ['direct_answer', 'search_required', 'tool_use', 'clarification']:
            results = hm.retrieve("hello AI", strategy, top_k=5)
            assert 'episodic' in results
            assert 'semantic' in results
            assert 'procedural' in results

    def test_store_lesson(self):
        from memabra.memory import HierarchicalMemory

        hm = HierarchicalMemory(embedding_fn=make_embedder())
        hm.store_lesson(
            problem="user asked about weather",
            failed_strategy="direct_answer",
            user_feedback="wrong city",
            better_approach="ask which city first"
        )
        assert len(hm.episodic.memories) == 1

    def test_save_and_load(self, tmp_path):
        from memabra.memory import HierarchicalMemory

        embed = make_embedder()
        hm = HierarchicalMemory(embedding_fn=embed)

        hm.episodic.add_interaction("q1", "a1", "direct_answer")
        hm.semantic.add_fact("Python", "has", "GIL")
        hm.procedural.add_skill("debug", ["error", "bug"], "Check logs")

        path = str(tmp_path / "memories.json")
        hm.save_to_disk(path)

        # Load into new instance
        hm2 = HierarchicalMemory(embedding_fn=embed)
        hm2.load_from_disk(path)

        assert len(hm2.episodic.memories) == 1
        assert len(hm2.semantic.memories) == 1
        assert len(hm2.procedural.memories) == 1


class TestActionStore:
    """Tests for ActionStore (broad-sense memory)."""

    def _sample_chain(self):
        return [
            {
                "step_index": 0,
                "action_type": "tool_call",
                "tool_or_skill": "search_file",
                "params": {"pattern": "*.py", "recursive": True},
                "result_summary": "found 5 files",
                "success": True,
                "latency_ms": 120,
                "timestamp": "2026-03-01T10:00:00",
            },
            {
                "step_index": 1,
                "action_type": "tool_call",
                "tool_or_skill": "read_file",
                "params": {"filePath": "tests/test_agent.py"},
                "result_summary": "read 119 lines",
                "success": True,
                "latency_ms": 30,
                "timestamp": "2026-03-01T10:00:01",
            },
        ]

    def test_record_action_chain(self):
        from memabra.memory import ActionStore

        store = ActionStore(embedding_fn=make_embedder())
        mid = store.record_action_chain(
            user_query="找到所有Python测试文件",
            strategy_used="tool_use",
            action_chain=self._sample_chain(),
            final_response_summary="找到5个测试文件",
            reward=0.9,
            context_tags=["file_search", "python"],
        )
        assert mid is not None
        mem = store.get(mid)
        assert mem.total_steps == 2
        assert set(mem.tools_used) == {"search_file", "read_file"}
        assert mem.total_latency_ms == 150
        assert mem.reward == 0.9

    def test_find_similar_chains(self):
        from memabra.memory import ActionStore

        store = ActionStore(embedding_fn=make_embedder())
        store.record_action_chain(
            user_query="搜索项目中的测试文件",
            strategy_used="tool_use",
            action_chain=self._sample_chain(),
            final_response_summary="找到文件",
            reward=0.8,
        )
        store.record_action_chain(
            user_query="做饭食谱",
            strategy_used="direct_answer",
            action_chain=[],
            final_response_summary="给出食谱",
            reward=0.5,
        )
        results = store.find_similar_chains("查找Python文件", top_k=2)
        assert len(results) <= 2

    def test_find_by_tool(self):
        from memabra.memory import ActionStore

        store = ActionStore(embedding_fn=make_embedder())
        store.record_action_chain(
            user_query="query1",
            strategy_used="tool_use",
            action_chain=self._sample_chain(),
            final_response_summary="done",
        )
        matches = store.find_by_tool("search_file")
        assert len(matches) == 1
        assert matches[0].user_query == "query1"

        no_matches = store.find_by_tool("nonexistent_tool")
        assert len(no_matches) == 0

    def test_find_successful_patterns(self):
        from memabra.memory import ActionStore

        store = ActionStore(embedding_fn=make_embedder())
        store.record_action_chain(
            user_query="good query",
            strategy_used="tool_use",
            action_chain=self._sample_chain(),
            final_response_summary="great result",
            reward=0.9,
            success=True,
        )
        store.record_action_chain(
            user_query="bad query",
            strategy_used="tool_use",
            action_chain=self._sample_chain(),
            final_response_summary="poor result",
            reward=0.1,
            success=False,
        )
        patterns = store.find_successful_patterns("search_file", min_reward=0.5)
        assert len(patterns) == 1
        assert patterns[0].user_query == "good query"

    def test_get_tool_stats(self):
        from memabra.memory import ActionStore

        store = ActionStore(embedding_fn=make_embedder())
        store.record_action_chain(
            user_query="q1",
            strategy_used="tool_use",
            action_chain=self._sample_chain(),
            final_response_summary="ok",
            reward=0.8,
        )
        stats = store.get_tool_stats()
        assert "search_file" in stats
        assert "read_file" in stats
        assert stats["search_file"]["total_calls"] == 1
        assert stats["search_file"]["success_rate"] == 1.0
        assert stats["search_file"]["avg_latency_ms"] == 120.0


class TestHierarchicalMemoryWithAction:
    """Tests for HierarchicalMemory action memory integration."""

    def test_retrieve_includes_action(self):
        from memabra.memory import HierarchicalMemory

        embed = make_embedder()
        hm = HierarchicalMemory(embedding_fn=embed)

        hm.action.record_action_chain(
            user_query="search for test files",
            strategy_used="tool_use",
            action_chain=[{
                "step_index": 0,
                "action_type": "tool_call",
                "tool_or_skill": "search_file",
                "params": {},
                "result_summary": "found files",
                "success": True,
                "latency_ms": 50,
            }],
            final_response_summary="found test files",
            reward=0.9,
        )

        for strategy in ['direct_answer', 'search_required', 'tool_use']:
            results = hm.retrieve("search test files", strategy, top_k=5)
            assert 'action' in results

    def test_tool_use_prioritizes_action(self):
        from memabra.memory import HierarchicalMemory

        embed = make_embedder()
        hm = HierarchicalMemory(embedding_fn=embed)

        hm.action.record_action_chain(
            user_query="find Python files",
            strategy_used="tool_use",
            action_chain=[{
                "step_index": 0,
                "action_type": "tool_call",
                "tool_or_skill": "search_file",
                "params": {"pattern": "*.py"},
                "result_summary": "found 10 files",
                "success": True,
                "latency_ms": 100,
            }],
            final_response_summary="listed Python files",
            reward=0.85,
        )

        results = hm.retrieve("find Python files", "tool_use", top_k=5)
        assert len(results['action']) > 0

    def test_save_and_load_with_action(self, tmp_path):
        from memabra.memory import HierarchicalMemory

        embed = make_embedder()
        hm = HierarchicalMemory(embedding_fn=embed)

        hm.action.record_action_chain(
            user_query="test query",
            strategy_used="tool_use",
            action_chain=[{
                "step_index": 0,
                "action_type": "tool_call",
                "tool_or_skill": "grep",
                "params": {"pattern": "TODO"},
                "result_summary": "3 matches",
                "success": True,
                "latency_ms": 45,
            }],
            final_response_summary="found 3 TODOs",
            reward=0.7,
            context_tags=["code_search"],
        )

        path = str(tmp_path / "memories.json")
        hm.save_to_disk(path)

        hm2 = HierarchicalMemory(embedding_fn=embed)
        hm2.load_from_disk(path)

        assert len(hm2.action.memories) == 1
        action_mem = list(hm2.action.memories.values())[0]
        assert action_mem.user_query == "test query"
        assert action_mem.tools_used == ["grep"]
        assert action_mem.reward == 0.7
        assert action_mem.context_tags == ["code_search"]
        assert len(action_mem.action_chain) == 1
