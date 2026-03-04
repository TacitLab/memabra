# 核心机制

## 1. 直觉网络 (Intuition Network)

### 1.1 设计哲学

直觉网络是 Memabra 的核心创新点。它模拟人类大脑的"系统1思维"——快速、自动、基于模式识别的决策。

关键特性：
- **快速**：单次前向传播，毫秒级响应
- **可学习**：随交互经验自我优化
- **可解释**：输出策略 ID，可追踪决策依据

### 1.2 轻量版实现 (MVP)

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleIntuitionNetwork:
    """
    基于余弦相似度的轻量直觉网络
    """
    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        # 策略库：每个策略对应一个原型向量
        self.strategies = {
            'direct_answer': np.random.randn(embedding_dim),
            'search_required': np.random.randn(embedding_dim),
            'tool_use': np.random.randn(embedding_dim),
            'clarification': np.random.randn(embedding_dim),
        }
        # 累积奖励，用于更新策略向量
        self.strategy_rewards = {k: [] for k in self.strategies}
    
    def predict(self, problem_embedding):
        """
        预测最佳策略
        
        Returns:
            strategy_id: 选择的策略
            confidence: 置信度 (最高相似度)
            similarities: 所有策略的相似度分数
        """
        problem_emb = np.array(problem_embedding).reshape(1, -1)
        
        similarities = {}
        for sid, strat_emb in self.strategies.items():
            sim = cosine_similarity(problem_emb, strat_emb.reshape(1, -1))[0][0]
            similarities[sid] = sim
        
        best_strategy = max(similarities, key=similarities.get)
        confidence = similarities[best_strategy]
        
        return best_strategy, confidence, similarities
    
    def update(self, problem_embedding, strategy_id, reward, lr=0.01):
        """
        基于奖励更新策略向量 (在线学习)
        
        Args:
            problem_embedding: 问题的嵌入
            strategy_id: 使用的策略
            reward: 反馈信号 (-1 到 +1)
            lr: 学习率
        """
        problem_emb = np.array(problem_embedding)
        current = self.strategies[strategy_id]
        
        # 向问题向量移动 (正奖励) 或远离 (负奖励)
        self.strategies[strategy_id] = current + lr * reward * problem_emb
        
        # 记录奖励历史
        self.strategy_rewards[strategy_id].append(reward)
```

### 1.3 进阶版实现 (神经网络)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralIntuitionNet(nn.Module):
    """
    小型神经网络直觉网络
    """
    def __init__(self, input_dim=384, hidden_dim=128, num_strategies=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_strategies),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
    
    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs
    
    def update(self, problem_emb, strategy_id, reward):
        """
        使用 REINFORCE 算法更新
        """
        self.optimizer.zero_grad()
        
        logits, probs = self.forward(problem_emb)
        log_prob = torch.log(probs[strategy_id])
        
        # REINFORCE: 最大化期望奖励
        loss = -log_prob * reward
        loss.backward()
        self.optimizer.step()
```

### 1.4 置信度阈值

```python
CONFIDENCE_THRESHOLD = 0.7

def decide_path(strategy_id, confidence):
    if confidence >= CONFIDENCE_THRESHOLD:
        return 'fast_path', strategy_id
    else:
        return 'exploration_mode', None
```

---

## 2. 记忆检索机制

### 2.1 分层检索策略

```python
class HierarchicalMemory:
    """
    三层记忆检索
    """
    def __init__(self, embedding_fn):
        self.episodic = EpisodicStore()      # 情景记忆
        self.semantic = SemanticStore()      # 语义记忆
        self.procedural = ProceduralStore()  # 程序记忆
        self.embed = embedding_fn
    
    def retrieve(self, problem_text, strategy_id, top_k=5):
        """
        基于策略选择检索方式
        """
        query_emb = self.embed(problem_text)
        
        if strategy_id == 'direct_answer':
            # 优先检索语义记忆（事实）
            return self.semantic.search(query_emb, top_k)
        
        elif strategy_id == 'search_required':
            # 检索情景记忆（类似过往对话）
            return self.episodic.search(query_emb, top_k)
        
        elif strategy_id == 'tool_use':
            # 检索程序记忆（技能、工具使用方法）
            return self.procedural.search(query_emb, top_k)
        
        elif strategy_id == 'clarification':
            # 检索上下文相关的追问模式
            return self.episodic.get_recent_context(top_k)
```

### 2.2 记忆存储格式

**情景记忆**：
```json
{
  "id": "uuid",
  "timestamp": "2026-03-04T14:30:00Z",
  "type": "interaction",
  "content": "用户询问股票代码",
  "embedding": [0.1, 0.2, ...],
  "outcome": "success",
  "strategy_used": "direct_answer"
}
```

**语义记忆**：
```json
{
  "id": "uuid",
  "type": "fact",
  "subject": "和顺石油",
  "predicate": "股票代码",
  "object": "603353",
  "embedding": [0.1, 0.2, ...],
  "source": "MEMORY.md",
  "confidence": 0.95
}
```

**程序记忆**：
```json
{
  "id": "uuid",
  "type": "skill",
  "name": "weather_query",
  "trigger_patterns": ["天气", "温度", "下雨"],
  "action": "call_weather_api",
  "success_rate": 0.92,
  "avg_reward": 0.85
}
```

---

## 3. 执行循环

### 3.1 主循环流程

```python
class MemabraAgent:
    def __init__(self):
        self.intuition = SimpleIntuitionNetwork()
        self.memory = HierarchicalMemory(embedding_fn)
        self.executor = Executor()
        self.evaluator = ImplicitEvaluator()
        self.history = []
    
    async def process(self, user_input, conversation_context):
        # 1. 编码问题
        problem_emb = embed(user_input)
        
        # 2. 直觉判断
        strategy_id, confidence, similarities = self.intuition.predict(problem_emb)
        
        # 3. 决定路径
        if confidence >= 0.7:
            # 快速路径
            memories = self.memory.retrieve(user_input, strategy_id)
            result = await self.executor.execute(strategy_id, memories, user_input)
        else:
            # 探索模式：尝试多种策略
            result = await self.exploration_mode(user_input, problem_emb)
        
        # 4. 记录本次交互
        interaction = {
            'input': user_input,
            'strategy': strategy_id,
            'confidence': confidence,
            'memories_used': memories,
            'output': result,
            'timestamp': now()
        }
        self.history.append(interaction)
        
        return result
    
    async def exploration_mode(self, user_input, problem_emb):
        """
        低置信度时，尝试多种策略并选择最佳
        """
        strategies_to_try = ['direct_answer', 'search_required', 'tool_use']
        candidates = []
        
        for sid in strategies_to_try:
            memories = self.memory.retrieve(user_input, sid)
            result = await self.executor.execute(sid, memories, user_input)
            
            # 启发式评估（快速但不精确）
            score = self.heuristic_score(result)
            candidates.append((sid, result, score))
        
        # 选择最佳结果
        best = max(candidates, key=lambda x: x[2])
        return best[1]
    
    def on_followup(self, user_response):
        """
        收到用户后续消息时，评估上一条回复
        """
        if not self.history:
            return
        
        last_interaction = self.history[-1]
        reward = self.evaluator.evaluate(
            last_interaction['output'], 
            user_response,
            conversation_context
        )
        
        # 更新直觉网络
        self.intuition.update(
            embed(last_interaction['input']),
            last_interaction['strategy'],
            reward
        )
        
        # 负反馈时触发深度搜索
        if reward < 0:
            self.deep_search_and_learn(last_interaction, user_response)
    
    def deep_search_and_learn(self, failed_interaction, user_feedback):
        """
        负反馈后的补救学习
        """
        # 1. 扩大记忆搜索范围
        extended_memories = self.memory.retrieve(
            failed_interaction['input'],
            failed_interaction['strategy'],
            top_k=20  # 扩大搜索
        )
        
        # 2. 尝试新策略
        alternative_strategy = self.select_alternative_strategy(failed_interaction)
        
        # 3. 记录这次"教训"
        self.memory.store_lesson(
            problem=failed_interaction['input'],
            failed_strategy=failed_interaction['strategy'],
            user_feedback=user_feedback,
            better_approach=alternative_strategy
        )
```

---

## 4. 关键设计决策

### 4.1 为什么用策略 ID 而非直接输出动作？

- **可解释性**：知道 Agent "想" 用什么方式解决问题
- **可学习性**：策略空间离散，便于强化学习
- **可组合**：策略可以组合（先澄清，再搜索，最后回答）

### 4.2 置信度阈值如何设定？

```python
# 动态阈值：基于历史表现自适应
class AdaptiveThreshold:
    def __init__(self, initial=0.7):
        self.threshold = initial
        self.history = []
    
    def update(self, confidence, success):
        self.history.append((confidence, success))
        
        # 如果高置信度经常失败，降低阈值
        # 如果低置信度经常成功，提高阈值
        if len(self.history) > 100:
            self.adjust_threshold()
```

### 4.3 记忆如何遗忘？

```python
def memory_decay(memory, current_time):
    """
    模拟艾宾浩斯遗忘曲线
    """
    age = current_time - memory['timestamp']
    base_strength = memory.get('strength', 1.0)
    
    # 艾宾浩斯遗忘函数
    retention = base_strength * exp(-age / (86400 * 5))  # 5天半衰期
    
    # 访问强化
    if memory.get('last_accessed'):
        access_boost = 0.1 * (1 + memory.get('access_count', 0))
        retention += access_boost
    
    return retention
```
