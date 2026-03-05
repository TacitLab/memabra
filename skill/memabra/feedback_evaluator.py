"""
Implicit Feedback System: Evaluate agent performance from user behavior.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from datetime import datetime, timedelta
import numpy as np


class SignalType(Enum):
    """反馈信号类型。"""
    EXPLICIT_POSITIVE = auto()    # 明确满意（谢谢/ok等）
    EXPLICIT_NEGATIVE = auto()    # 明确不满意（错了/不对等）
    IMPLICIT_REPEAT = auto()      # 重复追问
    IMPLICIT_REPHRASE = auto()    # 重述问题
    IMPLICIT_FOLLOWUP = auto()    # 追问细节
    IMPLICIT_TOPIC_SHIFT = auto() # 话题切换
    IMPLICIT_SILENCE = auto()     # 沉默/结束
    PARTIAL_SATISFACTION = auto() # 部分满意


@dataclass
class FeedbackSignal:
    """反馈信号。"""
    signal_type: SignalType
    reward: float          # -1.0 到 +1.0
    confidence: float      # 0.0 到 1.0，推断的确定性
    explanation: str       # 可解释性描述
    metadata: Dict = None  # 额外信息
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Interaction:
    """一次交互记录。"""
    id: str
    timestamp: datetime
    user_input: str
    assistant_output: str
    strategy_used: str
    confidence: float
    memories_used: List[str]
    feedback: Optional[FeedbackSignal] = None


class ImplicitEvaluator:
    """
    隐式反馈评估器：从用户自然行为中推断满意度。
    
    核心思想：不打扰用户，从对话流中读取信号。
    """
    
    # 明确正向关键词
    POSITIVE_KEYWORDS = {
        '谢谢', '感谢', '可以', 'ok', 'okay', '好的', '没错', '对', '正确',
        '完美', '搞定', '解决了', '明白了', '清楚', 'ok的', '对的',
        '好的谢谢', '可以了', '就这样', '没问题', '棒', '很好'
    }
    
    # 明确负向关键词
    NEGATIVE_KEYWORDS = {
        '不对', '错了', '不是', '不行', '没用', '不理解', '不清楚',
        '没解决', '不行啊', '不对啊', '不是这样的', '没懂', '还是不懂',
        '没用啊', '不行吧', '错的', '不对吧', '没明白', '依然不懂'
    }
    
    # 放弃信号
    GIVEUP_KEYWORDS = {
        '算了', '放弃', '不问了', '不管了', '就这样吧', '不折腾了',
        '不用了', '先这样', '随便吧', '无所谓了'
    }
    
    # 澄清追问词
    CLARIFICATION_PATTERNS = [
        '为什么', '怎么', '那', '还有', '另外', '以及', '而且',
        '具体', '详细', '举个例子', '比如说', '意思是'
    ]
    
    def __init__(
        self,
        embedding_fn=None,
        similarity_threshold: float = 0.75,
        repeat_threshold: float = 0.7
    ):
        self.embed = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.repeat_threshold = repeat_threshold
        
        # 统计
        self.stats = {
            'total_evaluations': 0,
            'signal_distribution': {t: 0 for t in SignalType},
            'avg_confidence': 0.0,
            'avg_reward': 0.0
        }
    
    def evaluate(
        self,
        last_assistant_msg: str,
        next_user_msg: str,
        conversation_history: List[Dict] = None,
        time_gap_seconds: Optional[float] = None
    ) -> FeedbackSignal:
        """
        评估上一条助理回复的质量。
        
        Args:
            last_assistant_msg: 助理上一条回复
            next_user_msg: 用户下一条消息
            conversation_history: 对话历史
            time_gap_seconds: 两条消息之间的时间间隔
            
        Returns:
            FeedbackSignal
        """
        self.stats['total_evaluations'] += 1
        
        user_msg = next_user_msg.strip().lower()
        
        # 1. 检查显式关键词（最高优先级）
        keyword_signal = self._check_explicit_keywords(user_msg)
        if keyword_signal:
            self._update_stats(keyword_signal)
            return keyword_signal
        
        # 2. 检查放弃信号
        if self._check_giveup(user_msg):
            signal = FeedbackSignal(
                signal_type=SignalType.EXPLICIT_NEGATIVE,
                reward=-0.8,
                confidence=0.85,
                explanation="用户表现出放弃或无奈情绪，表明问题未解决",
                metadata={'indicators': ['giveup_keywords']}
            )
            self._update_stats(signal)
            return signal
        
        # 3. 语义相似度分析（需要 embedding）
        if self.embed and last_assistant_msg:
            sim_score = self._semantic_similarity(last_assistant_msg, next_user_msg)
            
            # 高相似度 = 用户可能重复问（没解决）
            if sim_score > self.similarity_threshold:
                signal = FeedbackSignal(
                    signal_type=SignalType.IMPLICIT_REPEAT,
                    reward=-0.6,
                    confidence=min(0.9, sim_score),
                    explanation=f"用户消息与助理回复语义高度相似({sim_score:.2f})，可能表示重复追问",
                    metadata={'similarity': sim_score}
                )
                self._update_stats(signal)
                return signal
        
        # 4. 检查是否是重述问题
        if conversation_history and len(conversation_history) >= 2:
            prev_user_msgs = [
                turn['content'] for turn in conversation_history[-4:]
                if turn.get('role') == 'user'
            ]
            if prev_user_msgs:
                rephrase_score = self._check_rephrase(user_msg, prev_user_msgs[-1])
                if rephrase_score > 0.6:
                    signal = FeedbackSignal(
                        signal_type=SignalType.IMPLICIT_REPHRASE,
                        reward=-0.4,
                        confidence=rephrase_score,
                        explanation="用户似乎在重述问题，表明之前理解有偏差",
                        metadata={'rephrase_score': rephrase_score}
                    )
                    self._update_stats(signal)
                    return signal
        
        # 5. 分析追问意图
        intent = self._analyze_intent(user_msg)
        
        if intent == 'clarification':
            # 追问细节：部分满意但需要补充
            signal = FeedbackSignal(
                signal_type=SignalType.IMPLICIT_FOLLOWUP,
                reward=-0.15,
                confidence=0.6,
                explanation="用户追问细节，表明部分满意但需要更多信息",
                metadata={'intent': intent}
            )
            self._update_stats(signal)
            return signal
        
        elif intent == 'topic_shift':
            # 话题切换：结合上下文判断是否满意
            satisfaction = self._infer_satisfaction_from_context(
                conversation_history, time_gap_seconds
            )
            signal = FeedbackSignal(
                signal_type=SignalType.IMPLICIT_TOPIC_SHIFT,
                reward=satisfaction,
                confidence=0.5 if satisfaction == 0 else 0.6,
                explanation=f"用户切换话题，推断满意度为 {satisfaction:.2f}",
                metadata={'intent': intent, 'inferred_satisfaction': satisfaction}
            )
            self._update_stats(signal)
            return signal
        
        # 6. 默认中性
        signal = FeedbackSignal(
            signal_type=SignalType.IMPLICIT_SILENCE,
            reward=0.0,
            confidence=0.3,
            explanation="无法明确推断用户意图，视为中性",
            metadata={'intent': 'unknown'}
        )
        self._update_stats(signal)
        return signal
    
    def _check_explicit_keywords(self, msg: str) -> Optional[FeedbackSignal]:
        """检查显式关键词。"""
        msg_lower = msg.lower()
        
        # 检查正向关键词
        for kw in self.POSITIVE_KEYWORDS:
            if kw in msg_lower:
                return FeedbackSignal(
                    signal_type=SignalType.EXPLICIT_POSITIVE,
                    reward=+1.0,
                    confidence=0.9,
                    explanation=f"检测到正向关键词: '{kw}'",
                    metadata={'keyword': kw}
                )
        
        # 检查负向关键词
        for kw in self.NEGATIVE_KEYWORDS:
            if kw in msg_lower:
                return FeedbackSignal(
                    signal_type=SignalType.EXPLICIT_NEGATIVE,
                    reward=-1.0,
                    confidence=0.9,
                    explanation=f"检测到负向关键词: '{kw}'",
                    metadata={'keyword': kw}
                )
        
        return None
    
    def _check_giveup(self, msg: str) -> bool:
        """检查是否包含放弃信号。"""
        msg_lower = msg.lower()
        return any(kw in msg_lower for kw in self.GIVEUP_KEYWORDS)
    
    def _semantic_similarity(self, msg1: str, msg2: str) -> float:
        """计算语义相似度。"""
        if not self.embed:
            return 0.0
        
        try:
            emb1 = self.embed(msg1)
            emb2 = self.embed(msg2)
            
            # 余弦相似度
            dot = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot / (norm1 * norm2))
        except Exception:
            return 0.0
    
    def _check_rephrase(self, current_msg: str, previous_msg: str) -> float:
        """
        检查当前消息是否是之前消息的重述。
        
        Returns:
            0-1 分数，越高越可能是重述
        """
        if not self.embed:
            # 简单的字符串相似度
            from difflib import SequenceMatcher
            return SequenceMatcher(None, current_msg, previous_msg).ratio()
        
        # 语义相似度但内容略有不同
        sim = self._semantic_similarity(current_msg, previous_msg)
        
        # 编辑距离
        from difflib import SequenceMatcher
        string_sim = SequenceMatcher(None, current_msg, previous_msg).ratio()
        
        # 重述的特征：语义相似但措辞不同
        if sim > 0.6 and string_sim < 0.8:
            return (sim + (1 - string_sim)) / 2
        
        return 0.0
    
    def _analyze_intent(self, msg: str) -> str:
        """分析用户意图。"""
        msg_lower = msg.lower()
        
        # 检查澄清模式
        for pattern in self.CLARIFICATION_PATTERNS:
            if pattern in msg_lower:
                return 'clarification'
        
        # 简单启发：短消息可能是确认，长消息可能是追问
        if len(msg) < 10:
            return 'short_response'
        
        # 默认话题切换
        return 'topic_shift'
    
    def _infer_satisfaction_from_context(
        self,
        conversation_history: Optional[List[Dict]],
        time_gap_seconds: Optional[float]
    ) -> float:
        """
        从上下文推断满意度。
        
        话题切换时：
        - 如果对话已经很深入（多轮），可能是满意后换话题
        - 如果时间间隔很长，可能是放弃
        - 如果很短，可能是满意后自然切换
        """
        if not conversation_history:
            return 0.0
        
        turns = len([t for t in conversation_history if t.get('role') == 'user'])
        
        # 多轮对话后换话题：可能是满意
        if turns >= 3:
            return 0.3
        
        # 短时间切换：可能是满意
        if time_gap_seconds and time_gap_seconds < 60:
            return 0.2
        
        # 长时间无响应后换话题：可能是放弃
        if time_gap_seconds and time_gap_seconds > 300:  # 5分钟
            return -0.3
        
        return 0.0
    
    def _update_stats(self, signal: FeedbackSignal):
        """更新统计信息。"""
        self.stats['signal_distribution'][signal.signal_type] += 1
        
        # 移动平均
        n = self.stats['total_evaluations']
        self.stats['avg_confidence'] = (
            self.stats['avg_confidence'] * (n - 1) + signal.confidence
        ) / n
        self.stats['avg_reward'] = (
            self.stats['avg_reward'] * (n - 1) + signal.reward
        ) / n
    
    def get_stats(self) -> Dict:
        """获取统计信息。"""
        return {
            'total_evaluations': self.stats['total_evaluations'],
            'avg_confidence': self.stats['avg_confidence'],
            'avg_reward': self.stats['avg_reward'],
            'signal_distribution': {
                k.name: v for k, v in self.stats['signal_distribution'].items()
            }
        }


class DelayedRewardAssigner:
    """
    延迟奖励分配器：对话结束后分配整体奖励。
    
    使用 n-step return 思想，让前面的交互也能获得奖励信号。
    """
    
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma  # 折扣因子
        self.pending_conversations: Dict[str, List[Interaction]] = {}
    
    def start_conversation(self, conversation_id: str):
        """开始追踪一个对话。"""
        self.pending_conversations[conversation_id] = []
    
    def add_interaction(
        self,
        conversation_id: str,
        interaction: Interaction
    ):
        """添加交互记录。"""
        if conversation_id not in self.pending_conversations:
            self.start_conversation(conversation_id)
        self.pending_conversations[conversation_id].append(interaction)
    
    def finalize_conversation(
        self,
        conversation_id: str,
        final_signal: Optional[FeedbackSignal] = None
    ) -> List[Tuple[Interaction, float]]:
        """
        结束对话，分配奖励。
        
        Returns:
            [(interaction, reward), ...]
        """
        if conversation_id not in self.pending_conversations:
            return []
        
        interactions = self.pending_conversations[conversation_id]
        if not interactions:
            return []
        
        # 判断整体对话结果
        if final_signal:
            final_success = final_signal.reward
        else:
            final_success = self._judge_conversation_success(interactions)
        
        # 为每个交互分配延迟奖励
        results = []
        n = len(interactions)
        
        for i, interaction in enumerate(interactions):
            if interaction.feedback:
                # 如果已有即时反馈，结合使用
                immediate = interaction.feedback.reward
                # 距离结束越近，最终成功的影响越大
                steps_to_end = n - i - 1
                delayed = final_success * (self.gamma ** steps_to_end)
                # 加权平均
                reward = 0.6 * delayed + 0.4 * immediate
            else:
                steps_to_end = n - i - 1
                reward = final_success * (self.gamma ** steps_to_end)
            
            results.append((interaction, reward))
        
        # 清理
        del self.pending_conversations[conversation_id]
        
        return results
    
    def _judge_conversation_success(self, interactions: List[Interaction]) -> float:
        """判断对话整体是否成功。"""
        if not interactions:
            return 0.0
        
        # 收集所有反馈
        rewards = []
        for inter in interactions:
            if inter.feedback:
                rewards.append(inter.feedback.reward)
        
        if not rewards:
            return 0.0
        
        # 加权：后面的交互权重更高
        weights = [self.gamma ** i for i in range(len(rewards))]
        weights.reverse()  # 后面的权重更高
        
        weighted_sum = sum(r * w for r, w in zip(rewards, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0


class FeedbackCalibrator:
    """
    反馈校准器：定期校准反馈信号，减少偏差。
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.predictions: List[Dict] = []
        self.outcomes: List[Dict] = []
    
    def log_prediction(
        self,
        interaction_id: str,
        predicted_reward: float,
        predicted_confidence: float,
        context: Dict
    ):
        """记录预测。"""
        self.predictions.append({
            'interaction_id': interaction_id,
            'predicted_reward': predicted_reward,
            'predicted_confidence': predicted_confidence,
            'context': context,
            'timestamp': datetime.now()
        })
        
        # 保持窗口大小
        if len(self.predictions) > self.window_size:
            self.predictions = self.predictions[-self.window_size:]
    
    def log_outcome(
        self,
        interaction_id: str,
        actual_reward: float,
        conversation_success: bool
    ):
        """记录实际结果。"""
        self.outcomes.append({
            'interaction_id': interaction_id,
            'actual_reward': actual_reward,
            'conversation_success': conversation_success,
            'timestamp': datetime.now()
        })
    
    def calibrate(self) -> Dict[str, float]:
        """
        分析预测准确度，返回校准建议。
        
        Returns:
            校准统计
        """
        if len(self.predictions) < 10 or len(self.outcomes) < 10:
            return {'status': 'insufficient_data'}
        
        # 匹配预测和结果
        matched = []
        pred_map = {p['interaction_id']: p for p in self.predictions}
        for outcome in self.outcomes:
            if outcome['interaction_id'] in pred_map:
                matched.append({
                    **pred_map[outcome['interaction_id']],
                    **outcome
                })
        
        if len(matched) < 10:
            return {'status': 'insufficient_matches'}
        
        # 计算偏差
        predicted = [m['predicted_reward'] for m in matched]
        actual = [m['actual_reward'] for m in matched]
        
        bias = np.mean(predicted) - np.mean(actual)
        mae = np.mean([abs(p - a) for p, a in zip(predicted, actual)])
        
        # 检测系统性偏差
        false_positive = sum(1 for m in matched 
                           if m['predicted_reward'] > 0.3 and m['actual_reward'] < 0)
        false_negative = sum(1 for m in matched 
                           if m['predicted_reward'] < -0.3 and m['actual_reward'] > 0)
        
        return {
            'status': 'calibrated',
            'sample_size': len(matched),
            'bias': bias,  # 正 = 过度乐观，负 = 过度悲观
            'mae': mae,
            'false_positive_rate': false_positive / len(matched),
            'false_negative_rate': false_negative / len(matched),
            'recommendation': self._generate_recommendation(bias, false_positive, false_negative)
        }
    
    def _generate_recommendation(
        self,
        bias: float,
        false_positive: int,
        false_negative: int
    ) -> str:
        """生成校准建议。"""
        if bias > 0.2:
            return "系统过度乐观，建议提高负向关键词权重或降低相似度阈值"
        elif bias < -0.2:
            return "系统过度悲观，建议降低负向关键词权重"
        elif false_positive > false_negative * 2:
            return "误报率过高，建议提高正向判断门槛"
        elif false_negative > false_positive * 2:
            return "漏报率过高，建议降低负向判断门槛"
        else:
            return "系统校准良好"
