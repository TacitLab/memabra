"""Tests for Feedback System."""

import numpy as np
import pytest


def make_embedder(dim=64):
    """Create a simple deterministic embedder for testing."""
    def embedder(text):
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(dim)
        return (vec / np.linalg.norm(vec)).tolist()
    return embedder


class TestImplicitEvaluatorSimple:
    """Tests for the simplified ImplicitEvaluator (feedback.py)."""

    def test_positive_keyword(self):
        from memabra.feedback import ImplicitEvaluator, SignalType

        ev = ImplicitEvaluator()
        signal = ev.evaluate(
            last_assistant_msg="Here is the answer",
            next_user_msg="谢谢"
        )
        assert signal.signal_type == SignalType.EXPLICIT_POSITIVE
        assert signal.reward > 0

    def test_negative_keyword(self):
        from memabra.feedback import ImplicitEvaluator, SignalType

        ev = ImplicitEvaluator()
        signal = ev.evaluate(
            last_assistant_msg="Here is the answer",
            next_user_msg="错了，没用"
        )
        assert signal.signal_type == SignalType.EXPLICIT_NEGATIVE
        assert signal.reward < 0

    def test_clarification_intent(self):
        from memabra.feedback import ImplicitEvaluator, SignalType

        ev = ImplicitEvaluator()
        signal = ev.evaluate(
            last_assistant_msg="Python is a language",
            next_user_msg="为什么要用Python呢"
        )
        assert signal.signal_type == SignalType.IMPLICIT_FOLLOWUP

    def test_neutral_default(self):
        from memabra.feedback import ImplicitEvaluator

        ev = ImplicitEvaluator()
        signal = ev.evaluate(
            last_assistant_msg="Here is info",
            next_user_msg="abcdefghij random words"
        )
        assert abs(signal.reward) <= 0.5


class TestImplicitEvaluatorFull:
    """Tests for the full ImplicitEvaluator (feedback_evaluator.py)."""

    def test_positive_keyword(self):
        from memabra.feedback_evaluator import ImplicitEvaluator, SignalType

        ev = ImplicitEvaluator(embedding_fn=make_embedder())
        signal = ev.evaluate(
            last_assistant_msg="Answer",
            next_user_msg="好的谢谢"
        )
        assert signal.signal_type == SignalType.EXPLICIT_POSITIVE
        assert signal.reward == 1.0

    def test_giveup_detection(self):
        from memabra.feedback_evaluator import ImplicitEvaluator, SignalType

        ev = ImplicitEvaluator()
        signal = ev.evaluate(
            last_assistant_msg="Answer",
            next_user_msg="算了不问了"
        )
        assert signal.signal_type == SignalType.EXPLICIT_NEGATIVE
        assert signal.reward < 0

    def test_stats_tracking(self):
        from memabra.feedback_evaluator import ImplicitEvaluator

        ev = ImplicitEvaluator()
        ev.evaluate("msg", "谢谢")
        ev.evaluate("msg", "不对")

        stats = ev.get_stats()
        assert stats['total_evaluations'] == 2


class TestDelayedRewardAssigner:
    """Tests for DelayedRewardAssigner."""

    def test_assign_rewards_simple(self):
        from memabra.feedback import DelayedRewardAssigner

        assigner = DelayedRewardAssigner(gamma=0.9)
        conversation = [
            {'role': 'user', 'content': '帮我查天气'},
            {'role': 'assistant', 'content': '今天晴天'},
            {'role': 'user', 'content': '谢谢'},
        ]

        rewards = assigner.assign_rewards(conversation)
        assert len(rewards) == 1  # Only assistant turns
        _, reward = rewards[0]
        assert reward > 0, "Positive ending should yield positive reward"

    def test_negative_ending(self):
        from memabra.feedback import DelayedRewardAssigner

        assigner = DelayedRewardAssigner(gamma=0.9)
        conversation = [
            {'role': 'user', 'content': '帮我查股票'},
            {'role': 'assistant', 'content': '股票涨了'},
            {'role': 'user', 'content': '算了没用'},
        ]

        rewards = assigner.assign_rewards(conversation)
        _, reward = rewards[0]
        assert reward < 0, "Negative ending should yield negative reward"

    def test_empty_conversation(self):
        from memabra.feedback import DelayedRewardAssigner

        assigner = DelayedRewardAssigner()
        rewards = assigner.assign_rewards([])
        assert rewards == []


class TestFeedbackCalibrator:
    """Tests for FeedbackCalibrator."""

    def test_insufficient_data(self):
        from memabra.feedback_evaluator import FeedbackCalibrator

        cal = FeedbackCalibrator()
        result = cal.calibrate()
        assert result['status'] == 'insufficient_data'

    def test_calibration_with_data(self):
        from memabra.feedback_evaluator import FeedbackCalibrator

        cal = FeedbackCalibrator()

        for i in range(15):
            cal.log_prediction(
                interaction_id=f"int_{i}",
                predicted_reward=0.5,
                predicted_confidence=0.8,
                context={}
            )
            cal.log_outcome(
                interaction_id=f"int_{i}",
                actual_reward=0.3,
                conversation_success=True
            )

        result = cal.calibrate()
        assert result['status'] == 'calibrated'
        assert 'bias' in result
        assert 'mae' in result
