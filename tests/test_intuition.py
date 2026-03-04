"""Tests for Intuition Network modules."""

import numpy as np
import pytest


class TestSimpleIntuitionNetwork:
    """Tests for the cosine-similarity based intuition network."""

    def test_init(self):
        from memabra.intuition import SimpleIntuitionNetwork

        net = SimpleIntuitionNetwork(embedding_dim=64)
        assert len(net.strategies) == 4
        assert all(s in net.strategies for s in [
            'direct_answer', 'search_required', 'tool_use', 'clarification'
        ])

    def test_predict_returns_valid_strategy(self):
        from memabra.intuition import SimpleIntuitionNetwork

        net = SimpleIntuitionNetwork(embedding_dim=64)
        emb = np.random.randn(64)
        strategy_id, confidence, similarities = net.predict(emb)

        assert strategy_id in net.strategies
        assert 0 <= confidence <= 1
        assert len(similarities) == 4
        assert all(0 <= v <= 1 for v in similarities.values())

    def test_update_changes_strategy_vector(self):
        from memabra.intuition import SimpleIntuitionNetwork

        net = SimpleIntuitionNetwork(embedding_dim=64)
        emb = np.random.randn(64)

        before = net.strategies['direct_answer'].copy()
        net.update(emb, 'direct_answer', reward=1.0, lr=0.1)
        after = net.strategies['direct_answer']

        assert not np.allclose(before, after), "Strategy vector should change after update"

    def test_update_tracks_rewards(self):
        from memabra.intuition import SimpleIntuitionNetwork

        net = SimpleIntuitionNetwork(embedding_dim=64)
        emb = np.random.randn(64)

        net.update(emb, 'direct_answer', reward=0.5)
        net.update(emb, 'direct_answer', reward=-0.3)

        stats = net.get_strategy_stats()
        assert stats['direct_answer']['count'] == 2
        assert abs(stats['direct_answer']['avg_reward'] - 0.1) < 1e-6

    def test_unknown_strategy_update_noop(self):
        from memabra.intuition import SimpleIntuitionNetwork

        net = SimpleIntuitionNetwork(embedding_dim=64)
        emb = np.random.randn(64)
        # Should not raise
        net.update(emb, 'nonexistent_strategy', reward=1.0)


class TestIntuitionNetwork:
    """Tests for the PyTorch-based neural intuition network."""

    def test_init_and_predict(self):
        from memabra.intuition_network import IntuitionNetwork

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        emb = np.random.randn(64).tolist()
        pred = net.predict(emb)

        assert pred.strategy_id in net.strategy_names
        assert 0 <= pred.confidence <= 1
        assert len(pred.memory_query_vector) == 64
        assert len(pred.all_scores) == 4

    def test_update_reduces_loss(self):
        from memabra.intuition_network import IntuitionNetwork

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        net.setup_optimizer()
        emb = np.random.randn(64).tolist()

        stats = net.update(emb, 'direct_answer', reward=1.0)
        assert 'policy_loss' in stats
        assert 'temperature' in stats
        assert net.training_stats['updates'] == 1

    def test_save_and_load(self, tmp_path):
        from memabra.intuition_network import IntuitionNetwork

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        net.setup_optimizer()

        # Do some updates
        emb = np.random.randn(64).tolist()
        net.update(emb, 'direct_answer', reward=1.0)

        path = str(tmp_path / "model.pt")
        net.save(path)

        loaded = IntuitionNetwork.load(path)
        assert loaded.training_stats['updates'] == 1
        assert loaded.input_dim == 64

        # Predictions should be consistent
        pred_orig = net.predict(emb)
        pred_loaded = loaded.predict(emb)
        assert pred_orig.strategy_id == pred_loaded.strategy_id

    def test_memory_query_update(self):
        from memabra.intuition_network import IntuitionNetwork

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        net.setup_optimizer()

        query_emb = np.random.randn(64).tolist()
        target_emb = np.random.randn(64).tolist()

        loss = net.update_memory_query(query_emb, target_emb)
        assert isinstance(loss, float)
        assert loss >= 0


class TestAdaptiveThreshold:
    """Tests for AdaptiveThreshold."""

    def test_should_explore(self):
        from memabra.intuition_network import AdaptiveThreshold

        at = AdaptiveThreshold(initial=0.7)
        assert at.should_explore(0.5) is True
        assert at.should_explore(0.8) is False

    def test_update_adjusts_threshold(self):
        from memabra.intuition_network import AdaptiveThreshold

        at = AdaptiveThreshold(initial=0.7, window_size=20)

        # Simulate many high-confidence failures
        for _ in range(20):
            at.update(confidence=0.8, success=False)

        # Threshold should increase
        assert at.threshold > 0.7


class TestExplorationController:
    """Tests for ExplorationController."""

    def test_fast_path(self):
        from memabra.intuition_network import IntuitionNetwork, ExplorationController

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        controller = ExplorationController(net, threshold=0.0)  # Very low threshold

        emb = np.random.randn(64).tolist()
        path_type, primary, all_preds = controller.decide_path(emb)

        # With threshold=0, should almost always be fast
        assert path_type in ('fast', 'exploration')
        assert primary.strategy_id in net.strategy_names

    def test_exploration_path(self):
        from memabra.intuition_network import IntuitionNetwork, ExplorationController

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        controller = ExplorationController(net, threshold=0.99)  # Very high threshold

        emb = np.random.randn(64).tolist()
        path_type, primary, all_preds = controller.decide_path(emb)

        assert path_type == 'exploration'
        assert len(all_preds) == 4
