# 隐式反馈系统

## 核心理念

显式反馈（用户点击 👍/👎）破坏交互流畅性，且收集困难。隐式反馈从用户的**自然行为**中推断满意度，是更符合 Agent 交互体验的设计。

## 信号来源

### 1. 即时信号（下一条消息）

| 用户行为 | 推断含义 | 奖励值 | 置信度 |
|----------|----------|--------|--------|
| "谢谢"/"可以了"/"ok" | 明确满意 | +1.0 | 高 |
| "不对"/"错了"/"不是" | 明确不满意 | -1.0 | 高 |
| 追问同一问题（语义相似 > 0.8） | 未解决 | -0.5 | 高 |
| 重述/修改问题 | 理解偏差 | -0.4 | 中 |
| 追问相关细节 | 部分满意，需补充 | -0.1 | 中 |
| 转向新话题 | 模糊（可能满意/放弃） | 0.0 | 低 |

### 2. 短期信号（对话流）

| 模式 | 推断含义 | 处理方式 |
|------|----------|----------|
| 连续追问同一主题 | 持续不满意 | 累计负向，触发策略切换 |
| 追问后问题解决 | 最终满意 | 前面负向奖励修正 |
| 长时间沉默后追问 | 结果复杂/不满意 | 轻微负向 |
| 对话自然结束 | 任务完成 | 整体正向 |

### 3. 长期信号（跨会话）

| 指标 | 含义 |
|------|------|
| 记忆被再次调用频率 | 有效性指标 |
| 关联记忆的成功率 | 知识图谱质量 |
| 策略的长期胜率 | 直觉网络校准度 |

## 实现设计

### 核心类

```python
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

class SignalType(Enum):
    EXPLICIT_POSITIVE = "explicit_positive"    # "谢谢"
    EXPLICIT_NEGATIVE = "explicit_negative"    # "错了"
    IMPLICIT_REPEAT = "implicit_repeat"        # 重复追问
    IMPLICIT_REPHRASE = "implicit_rephrase"    # 重述问题
    IMPLICIT_FOLLOWUP = "implicit_followup"    # 追问细节
    IMPLICIT_TOPIC_SHIFT = "topic_shift"       # 换话题
    IMPLICIT_SILENCE = "silence"               # 沉默/结束

@dataclass
class FeedbackSignal:
    signal_type: SignalType
    reward: float
    confidence: float  # 0-1，表示推断的确定性
    explanation: str   # 可解释性

class ImplicitEvaluator:
    """
    隐式反馈评估器
    """
    
    # 关键词映射
    POSITIVE_KEYWORDS = {'谢谢', '可以', 'ok', '好的', '没错', '对', '完美'}
    NEGATIVE_KEYWORDS = {'不对', '错了', '不是', '不行', '没用', '不理解'}
    
    def __init__(self, embedding_fn):
        self.embed = embedding_fn
        self.similarity_threshold = 0.75
    
    def evaluate(self, 
                 last_assistant_msg: str, 
                 next_user_msg: str,
                 conversation_history: List[dict] = None) -> FeedbackSignal:
        """
        基于用户后续行为评估上一条回复
        """
        
        # 1. 检查显式关键词
        keyword_signal = self._check_keywords(next_user_msg)
        if keyword_signal:
            return keyword_signal
        
        # 2. 语义相似度分析
        sim_score = self._semantic_similarity(last_assistant_msg, next_user_msg)
        
        if sim_score > self.similarity_threshold:
            # 用户在说类似的内容 = 之前没解决
            return FeedbackSignal(
                signal_type=SignalType.IMPLICIT_REPEAT,
                reward=-0.5,
                confidence=0.8,
                explanation=f"用户重复类似问题 (相似度: {sim_score:.2f})，表明之前回复未解决需求"
            )
        
        # 3. 分析追问意图
        intent = self._analyze_intent(next_user_msg)
        
        if intent == 'clarification':
            return FeedbackSignal(
                signal_type=SignalType.IMPLICIT_FOLLOWUP,
                reward=-0.1,
                confidence=0.6,
                explanation="用户追问细节，表明部分满意但需补充"
            )
        
        elif intent == 'topic_shift':
            # 需要结合上下文判断是否满意
            satisfaction = self._infer_satisfaction_from_context(conversation_history)
            return FeedbackSignal(
                signal_type=SignalType.IMPLICIT_TOPIC_SHIFT,
                reward=satisfaction,
                confidence=0.4,
                explanation="用户切换话题，满意度需结合上下文推断"
            )
        
        # 4. 默认中性
        return FeedbackSignal(
            signal_type=SignalType.IMPLICIT_SILENCE,
            reward=0.0,
            confidence=0.3,
            explanation="无法明确推断用户意图"
        )
    
    def _check_keywords(self, msg: str) -> Optional[FeedbackSignal]:
        msg_lower = msg.lower()
        
        for kw in self.POSITIVE_KEYWORDS:
            if kw in msg_lower:
                return FeedbackSignal(
                    signal_type=SignalType.EXPLICIT_POSITIVE,
                    reward=+1.0,
                    confidence=0.9,
                    explanation=f"检测到正向关键词: '{kw}'"
                )
        
        for kw in self.NEGATIVE_KEYWORDS:
            if kw in msg_lower:
                return FeedbackSignal(
                    signal_type=SignalType.EXPLICIT_NEGATIVE,
                    reward=-1.0,
                    confidence=0.9,
                    explanation=f"检测到负向关键词: '{kw}'"
                )
        
        return None
    
    def _semantic_similarity(self, msg1: str, msg2: str) -> float:
        emb1 = self.embed(msg1)
        emb2 = self.embed(msg2)
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def _analyze_intent(self, msg: str) -> str:
        """
        简单规则判断用户意图
        """
        clarification_patterns = ['为什么', '怎么', '那', '还有', '另外']
        
        for pattern in clarification_patterns:
            if pattern in msg:
                return 'clarification'
        
        # 如果与历史主题差异大，视为话题切换
        return 'topic_shift'
```

### 延迟奖励分配

```python
class DelayedRewardAssigner:
    """
    处理延迟反馈：对话结束后再分配奖励
    """
    
    def __init__(self, gamma=0.9):
        self.gamma = gamma  # 折扣因子
    
    def assign_rewards(self, conversation: List[dict]) -> List[Tuple[dict, float]]:
        """
        为对话中的每个交互分配奖励
        
        Args:
            conversation: 完整对话历史
            
        Returns:
            [(interaction, reward), ...]
        """
        n = len(conversation)
        rewards = []
        
        # 判断整体对话是否成功
        final_success = self._judge_conversation_success(conversation)
        
        # n-step return: 后面的交互受前面影响
        for i, turn in enumerate(conversation):
            if turn['role'] == 'assistant':
                # 距离结束越近，权重越高
                steps_to_end = n - i
                reward = final_success * (self.gamma ** steps_to_end)
                
                # 如果有即时负反馈，叠加
                if turn.get('immediate_feedback'):
                    reward = 0.7 * reward + 0.3 * turn['immediate_feedback']
                
                rewards.append((turn, reward))
        
        return rewards
    
    def _judge_conversation_success(self, conversation: List[dict]) -> float:
        """
        判断整个对话是否成功
        """
        # 简单启发式：
        # - 以感谢结束 = 成功
        # - 以放弃/长时间沉默结束 = 失败
        # - 其他 = 中性
        
        last_user_msgs = [t for t in conversation[-3:] if t['role'] == 'user']
        if not last_user_msgs:
            return 0.0
        
        last_msg = last_user_msgs[-1]['content']
        
        # 正向结束信号
        if any(kw in last_msg for kw in ['谢谢', 'ok', '可以', '搞定']):
            return +1.0
        
        # 负向结束信号
        if any(kw in last_msg for kw in ['算了', '放弃', '不问了']):
            return -1.0
        
        # 检查是否有多次重复追问（失败迹象）
        repeat_count = self._count_repeats(conversation)
        if repeat_count >= 3:
            return -0.5
        
        return 0.3  # 默认轻微正向
    
    def _count_repeats(self, conversation: List[dict]) -> int:
        """统计用户重复追问的次数"""
        user_msgs = [t['content'] for t in conversation if t['role'] == 'user']
        
        repeats = 0
        for i in range(1, len(user_msgs)):
            sim = self._semantic_similarity(user_msgs[i-1], user_msgs[i])
            if sim > 0.7:
                repeats += 1
        
        return repeats
```

## 反例处理

### 误判场景

| 场景 | 误判 | 解决方案 |
|------|------|----------|
| 用户说"不对，我是问..." | 高负向 | 检测重述模式，降低惩罚 |
| 用户只是礼貌回复"谢谢"后换话题 | 高正向 | 结合会话长度判断 |
| 复杂问题需要多轮澄清 | 累计负向过高 | 检测"逐步收敛"模式 |

### 校准机制

```python
class FeedbackCalibrator:
    """
    定期校准反馈信号，减少偏差
    """
    
    def __init__(self):
        self.predicted_rewards = []
        self.actual_outcomes = []
    
    def log_prediction(self, predicted_reward, context):
        self.predicted_rewards.append({
            'reward': predicted_reward,
            'context': context,
            'timestamp': now()
        })
    
    def log_outcome(self, conversation_id, actual_success):
        # 对话结束后，对比预测和实际
        self.actual_outcomes.append({
            'conversation_id': conversation_id,
            'success': actual_success
        })
    
    def calibrate(self):
        """
        分析预测准确度，调整阈值
        """
        # 如果预测负向但实际成功，降低负向权重
        # 如果预测正向但实际失败，提高阈值
        pass
```

## 配置示例

```yaml
# config/feedback.yaml
implicit_feedback:
  enabled: true
  
  # 关键词配置
  positive_keywords:
    - "谢谢"
    - "可以"
    - "ok"
    - "搞定了"
    
  negative_keywords:
    - "不对"
    - "错了"
    - "不是"
    - "不行"
  
  # 相似度阈值
  similarity_threshold: 0.75
  
  # 奖励权重
  reward_weights:
    explicit: 1.0      # 显式关键词
    implicit: 0.7      # 隐式信号
    delayed: 0.5       # 延迟奖励
  
  # 延迟分配
  delayed_assignment:
    enabled: true
    gamma: 0.9         # 折扣因子
    min_turns: 3       # 最少轮数才触发
```
