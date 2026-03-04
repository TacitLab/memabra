"""Tests for MemabraAgent integration."""

import pytest


class TestMemabraAgent:
    """Integration tests for the full MemabraAgent."""

    def test_create_agent(self):
        from memabra.agent import MemabraAgent

        agent = MemabraAgent(embedding_dim=64)
        assert agent.embedding_dim == 64
        assert agent.stats['total_interactions'] == 0

    def test_process_returns_valid_result(self):
        from memabra.agent import MemabraAgent

        agent = MemabraAgent(embedding_dim=64)
        result = agent.process("今天天气怎么样")

        assert 'interaction_id' in result
        assert 'response' in result
        assert 'strategy' in result
        assert 'confidence' in result
        assert result['strategy'] in [
            'direct_answer', 'search_required', 'tool_use', 'clarification'
        ]
        assert agent.stats['total_interactions'] == 1

    def test_followup_feedback(self):
        from memabra.agent import MemabraAgent

        agent = MemabraAgent(embedding_dim=64)
        agent.process("帮我优化SQL")
        feedback = agent.on_user_followup("谢谢，速度快多了")

        assert feedback is not None
        assert 'feedback_type' in feedback
        assert 'reward' in feedback
        assert feedback['reward'] > 0, "Positive followup should give positive reward"

    def test_negative_followup(self):
        from memabra.agent import MemabraAgent

        agent = MemabraAgent(embedding_dim=64)
        agent.process("查一下股票")
        feedback = agent.on_user_followup("错了，没用")

        assert feedback is not None
        assert feedback['reward'] < 0

    def test_multiple_interactions(self):
        from memabra.agent import MemabraAgent

        agent = MemabraAgent(embedding_dim=64)

        for i in range(3):
            result = agent.process(f"问题 {i}")
            assert result is not None

        assert agent.stats['total_interactions'] == 3

    def test_update_from_feedback(self):
        from memabra.agent import MemabraAgent

        agent = MemabraAgent(embedding_dim=64)
        emb = agent.embedder("测试输入")

        stats = agent.update_from_feedback(
            query_embedding=emb,
            strategy_id='direct_answer',
            reward=0.8
        )

        assert 'loss' in stats
        assert 'total_updates' in stats
        assert stats['total_updates'] == 1

    def test_get_stats(self):
        from memabra.agent import MemabraAgent

        agent = MemabraAgent(embedding_dim=64)
        agent.process("test")

        stats = agent.get_stats()
        assert stats['total_interactions'] == 1
        assert 'intuition_stats' in stats
        assert 'evaluator_stats' in stats

    def test_save_and_load(self, tmp_path):
        from memabra.agent import MemabraAgent

        agent = MemabraAgent(embedding_dim=64)
        agent.process("记住这个")

        path = str(tmp_path / "model.pt")
        agent.save(path)

        # Load into new agent
        agent2 = MemabraAgent(embedding_dim=64, model_path=path)
        assert agent2.intuition.training_stats['updates'] == 0  # No feedback updates yet

    def test_reset_conversation(self):
        from memabra.agent import MemabraAgent

        agent = MemabraAgent(embedding_dim=64)
        agent.process("hello")
        agent.process("world")

        assert len(agent.conversation_history) == 4  # 2 user + 2 assistant
        agent.reset_conversation()
        assert len(agent.conversation_history) == 0

    def test_demo_runs_without_error(self):
        from memabra.agent import demo
        # Should complete without exceptions
        demo()
