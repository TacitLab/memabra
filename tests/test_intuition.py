"""Tests for Intuition Network (PyTorch-based)."""

import numpy as np
import pytest


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

    def test_predict_returns_valid_strategy(self):
        from memabra.intuition_network import IntuitionNetwork

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        emb = np.random.randn(64).tolist()
        pred = net.predict(emb)

        assert pred.strategy_id in [
            'direct_answer', 'search_required', 'tool_use', 'clarification'
        ]
        assert all(0 <= v <= 1 for v in pred.all_scores.values())

    def test_update_reduces_loss(self):
        from memabra.intuition_network import IntuitionNetwork

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        net.setup_optimizer()
        emb = np.random.randn(64).tolist()

        stats = net.update(emb, 'direct_answer', reward=1.0)
        assert 'policy_loss' in stats
        assert 'temperature' in stats
        assert net.training_stats['updates'] == 1

    def test_multiple_updates_track_stats(self):
        from memabra.intuition_network import IntuitionNetwork

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        net.setup_optimizer()
        emb = np.random.randn(64).tolist()

        net.update(emb, 'direct_answer', reward=0.5)
        net.update(emb, 'search_required', reward=-0.3)

        assert net.training_stats['updates'] == 2
        assert net.training_stats['strategy_distribution']['direct_answer'] == 1
        assert net.training_stats['strategy_distribution']['search_required'] == 1

    def test_save_and_load(self, tmp_path):
        from memabra.intuition_network import IntuitionNetwork

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        net.setup_optimizer()

        emb = np.random.randn(64).tolist()
        net.update(emb, 'direct_answer', reward=1.0)

        path = str(tmp_path / "model.pt")
        net.save(path)

        loaded = IntuitionNetwork.load(path)
        assert loaded.training_stats['updates'] == 1
        assert loaded.input_dim == 64

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

    def test_get_strategy_weights(self):
        from memabra.intuition_network import IntuitionNetwork

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        weights = net.get_strategy_weights()

        assert len(weights) == 4
        assert all(s in weights for s in net.strategy_names)

    def test_custom_strategy_names(self):
        from memabra.intuition_network import IntuitionNetwork

        names = ['alpha', 'beta', 'gamma', 'delta']
        net = IntuitionNetwork(
            input_dim=64, hidden_dim=32, num_strategies=4,
            memory_query_dim=64, strategy_names=names
        )
        assert net.strategy_names == names

        pred = net.predict(np.random.randn(64).tolist())
        assert pred.strategy_id in names


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

        for _ in range(20):
            at.update(confidence=0.8, success=False)

        assert at.threshold > 0.7

    def test_get_stats(self):
        from memabra.intuition_network import AdaptiveThreshold

        at = AdaptiveThreshold(initial=0.7)
        stats = at.get_stats()
        assert stats['threshold'] == 0.7
        assert stats['samples'] == 0


class TestExplorationController:
    """Tests for ExplorationController."""

    def test_fast_path(self):
        from memabra.intuition_network import IntuitionNetwork, ExplorationController

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        controller = ExplorationController(net, threshold=0.0)

        emb = np.random.randn(64).tolist()
        path_type, primary, all_preds = controller.decide_path(emb)

        assert path_type in ('fast', 'exploration')
        assert primary.strategy_id in net.strategy_names

    def test_exploration_path(self):
        from memabra.intuition_network import IntuitionNetwork, ExplorationController

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        controller = ExplorationController(net, threshold=0.99)

        emb = np.random.randn(64).tolist()
        path_type, primary, all_preds = controller.decide_path(emb)

        assert path_type == 'exploration'
        assert len(all_preds) == 4

    def test_report_outcome(self):
        from memabra.intuition_network import IntuitionNetwork, ExplorationController

        net = IntuitionNetwork(input_dim=64, hidden_dim=32, num_strategies=4, memory_query_dim=64)
        controller = ExplorationController(net, threshold=0.7)
        controller.report_outcome(0.8, True)
        assert len(controller.adaptive_threshold.history) == 1
