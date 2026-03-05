"""Tests for Feedback System (feedback_evaluator.py)."""

import numpy as np
import pytest


def make_embedder(dim=64):
    """Create a simple deterministic embedder for testing."""
    def embedder(text):
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(dim)
        return (vec / np.linalg.norm(vec)).tolist()
    return embedder


class TestImplicitEvaluator:
    """Tests for ImplicitEvaluator."""

    def test_positive_keyword(self):
        from memabra.feedback_evaluator import ImplicitEvaluator, SignalType

        ev = ImplicitEvaluator(embedding_fn=make_embedder())
        signal = ev.evaluate(
            last_assistant_msg="Answer",
            next_user_msg="好的谢谢"
        )
        assert signal.signal_type == SignalType.EXPLICIT_POSITIVE
        assert signal.reward == 1.0

    def test_negative_keyword(self):
        from memabra.feedback_evaluator import ImplicitEvaluator, SignalType

        ev = ImplicitEvaluator()
        signal = ev.evaluate(
            last_assistant_msg="Here is the answer",
            next_user_msg="错了，没用"
        )
        assert signal.signal_type == SignalType.EXPLICIT_NEGATIVE
        assert signal.reward < 0

    def test_giveup_detection(self):
        from memabra.feedback_evaluator import ImplicitEvaluator, SignalType

        ev = ImplicitEvaluator()
        signal = ev.evaluate(
            last_assistant_msg="Answer",
            next_user_msg="算了不问了"
        )
        assert signal.signal_type == SignalType.EXPLICIT_NEGATIVE
        assert signal.reward < 0

    def test_clarification_intent(self):
        from memabra.feedback_evaluator import ImplicitEvaluator, SignalType

        ev = ImplicitEvaluator()
        signal = ev.evaluate(
            last_assistant_msg="Python is a language",
            next_user_msg="为什么要用Python呢"
        )
        assert signal.signal_type == SignalType.IMPLICIT_FOLLOWUP

    def test_neutral_default(self):
        from memabra.feedback_evaluator import ImplicitEvaluator

        ev = ImplicitEvaluator()
        signal = ev.evaluate(
            last_assistant_msg="Here is info",
            next_user_msg="abcdefghij random words"
        )
        assert abs(signal.reward) <= 0.5

    def test_stats_tracking(self):
        from memabra.feedback_evaluator import ImplicitEvaluator

        ev = ImplicitEvaluator()
        ev.evaluate("msg", "谢谢")
        ev.evaluate("msg", "不对")

        stats = ev.get_stats()
        assert stats['total_evaluations'] == 2

    def test_semantic_similarity_repeat(self):
        from memabra.feedback_evaluator import ImplicitEvaluator, SignalType

        ev = ImplicitEvaluator(
            embedding_fn=make_embedder(),
            similarity_threshold=0.0  # very low to trigger repeat
        )
        signal = ev.evaluate(
            last_assistant_msg="some answer",
            next_user_msg="some answer"
        )
        # Should detect as repeat or explicit keyword
        assert signal.reward <= 0


class TestDelayedRewardAssigner:
    """Tests for DelayedRewardAssigner."""

    def test_start_and_finalize(self):
        from memabra.feedback_evaluator import DelayedRewardAssigner, Interaction, FeedbackSignal, SignalType
        from datetime import datetime

        assigner = DelayedRewardAssigner(gamma=0.9)
        assigner.start_conversation("conv1")

        interaction = Interaction(
            id="int1",
            timestamp=datetime.now(),
            user_input="帮我查天气",
            assistant_output="今天晴天",
            strategy_used="direct_answer",
            confidence=0.8,
            memories_used=[],
            feedback=FeedbackSignal(
                signal_type=SignalType.EXPLICIT_POSITIVE,
                reward=1.0,
                confidence=0.9,
                explanation="positive"
            )
        )
        assigner.add_interaction("conv1", interaction)

        results = assigner.finalize_conversation("conv1")
        assert len(results) == 1
        _, reward = results[0]
        assert reward > 0

    def test_empty_conversation(self):
        from memabra.feedback_evaluator import DelayedRewardAssigner

        assigner = DelayedRewardAssigner()
        results = assigner.finalize_conversation("nonexistent")
        assert results == []

    def test_gamma_discount(self):
        from memabra.feedback_evaluator import DelayedRewardAssigner, Interaction
        from datetime import datetime

        assigner = DelayedRewardAssigner(gamma=0.5)
        assigner.start_conversation("conv2")

        for i in range(3):
            interaction = Interaction(
                id=f"int{i}",
                timestamp=datetime.now(),
                user_input=f"q{i}",
                assistant_output=f"a{i}",
                strategy_used="direct_answer",
                confidence=0.8,
                memories_used=[]
            )
            assigner.add_interaction("conv2", interaction)

        from memabra.feedback_evaluator import FeedbackSignal, SignalType
        final_signal = FeedbackSignal(
            signal_type=SignalType.EXPLICIT_POSITIVE,
            reward=1.0,
            confidence=0.9,
            explanation="positive ending"
        )
        results = assigner.finalize_conversation("conv2", final_signal)
        assert len(results) == 3

        rewards = [r for _, r in results]
        # Earlier interactions should have lower discounted reward
        assert rewards[0] < rewards[2]


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
        assert 'recommendation' in result
