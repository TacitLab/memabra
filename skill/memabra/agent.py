"""
Memabra 核心集成示例

展示如何组合使用：
1. IntuitionNetwork - 策略选择 + 记忆查询向量生成
2. ImplicitEvaluator - 从用户行为推断反馈
3. HierarchicalMemory - 记忆存储和检索
4. DelayedRewardAssigner - 延迟奖励分配
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import json
import uuid

# 模拟 embedding 函数（实际使用时替换为 sentence-transformers）
class DummyEmbedder:
    """简单的模拟 embedding 生成器。"""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        np.random.seed(42)
        # 为常见词预定义一些向量
        self.word_vectors = {}
    
    def __call__(self, text: str) -> List[float]:
        """生成文本的 embedding。"""
        # 使用哈希来生成一致的向量
        words = text.lower().split()
        vec = np.zeros(self.dim)
        
        for word in words:
            if word not in self.word_vectors:
                # 基于词哈希生成一致向量
                np.random.seed(hash(word) % (2**32))
                self.word_vectors[word] = np.random.randn(self.dim)
            vec += self.word_vectors[word]
        
        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec.tolist()


class MemabraAgent:
    """
    Memabra 完整代理：集成直觉网络、记忆系统和反馈评估。
    
    使用流程：
    1. 接收用户输入
    2. IntuitionNetwork 选择策略 + 生成记忆查询向量
    3. 使用查询向量检索相关记忆
    4. 执行对应策略
    5. 记录交互，等待后续反馈
    6. 收到反馈后更新直觉网络
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        model_path: Optional[str] = None
    ):
        self.embedding_dim = embedding_dim
        self.embedder = DummyEmbedder(dim=embedding_dim)
        
        # 初始化直觉网络
        if model_path:
            from .intuition_network import IntuitionNetwork
            self.intuition = IntuitionNetwork.load(model_path)
        else:
            from .intuition_network import IntuitionNetwork
            self.intuition = IntuitionNetwork(
                input_dim=embedding_dim,
                hidden_dim=256,
                num_strategies=4,
                memory_query_dim=embedding_dim,
                strategy_names=['direct_answer', 'search_required', 'tool_use', 'clarification'],
                lr=1e-3
            )
            self.intuition.setup_optimizer()
        
        # 初始化反馈评估器
        from .feedback_evaluator import ImplicitEvaluator, DelayedRewardAssigner
        self.evaluator = ImplicitEvaluator(embedding_fn=self.embedder)
        self.reward_assigner = DelayedRewardAssigner(gamma=0.9)
        
        # 初始化记忆系统（简化版，使用内存）
        from .memory import HierarchicalMemory
        self.memory = HierarchicalMemory(embedding_fn=self.embedder)
        
        # 对话状态
        self.current_conversation_id: Optional[str] = None
        self.conversation_history: List[Dict] = []
        self.last_interaction_id: Optional[str] = None
        self._interaction_log: List[Dict] = []  # 交互记录，用于反馈学习
        
        # 统计
        self.stats = {
            'total_interactions': 0,
            'strategy_usage': {s: 0 for s in self.intuition.strategy_names},
            'total_reward': 0.0
        }
    
    def process(self, user_input: str) -> Dict:
        """
        处理用户输入，返回响应。
        
        这是主要的交互入口。
        """
        # 1. 生成输入的 embedding
        query_embedding = self.embedder(user_input)
        
        # 2. 直觉网络预测策略和记忆查询向量
        prediction = self.intuition.predict(query_embedding)
        
        # 3. 使用生成的记忆查询向量检索记忆
        memories = self.memory.retrieve(
            query_text=user_input,
            strategy_id=prediction.strategy_id,
            top_k=5
        )
        
        # 4. 根据策略执行（简化演示）
        response = self._execute_strategy(
            strategy_id=prediction.strategy_id,
            user_input=user_input,
            memories=memories,
            confidence=prediction.confidence
        )
        
        # 5. 记录交互
        interaction_id = str(uuid.uuid4())[:8]
        self.last_interaction_id = interaction_id
        
        interaction = {
            'id': interaction_id,
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'query_embedding': query_embedding,
            'strategy_used': prediction.strategy_id,
            'confidence': prediction.confidence,
            'all_strategy_scores': prediction.all_scores,
            'memory_query_vector': prediction.memory_query_vector,
            'memories_retrieved': len(memories.get('episodic', [])) + 
                                  len(memories.get('semantic', [])) + 
                                  len(memories.get('procedural', [])) +
                                  len(memories.get('action', [])),
            'assistant_response': response,
        }
        
        self._interaction_log.append(interaction)
        
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        self.stats['total_interactions'] += 1
        self.stats['strategy_usage'][prediction.strategy_id] += 1
        
        return {
            'interaction_id': interaction_id,
            'response': response,
            'strategy': prediction.strategy_id,
            'confidence': prediction.confidence,
            'memories_used': interaction['memories_retrieved']
        }
    
    def _execute_strategy(
        self,
        strategy_id: str,
        user_input: str,
        memories: Dict,
        confidence: float
    ) -> str:
        """
        根据策略执行动作（简化演示）。
        
        实际应用中，这里会调用真实的工具或生成逻辑。
        """
        if strategy_id == 'direct_answer':
            return f"[Direct Answer] 基于记忆直接回复：'{user_input[:30]}...' (置信度: {confidence:.2f})"
        
        elif strategy_id == 'search_required':
            return f"[Search] 需要搜索来回答：'{user_input[:30]}...'"
        
        elif strategy_id == 'tool_use':
            return f"[Tool Use] 调用工具处理：'{user_input[:30]}...'"
        
        elif strategy_id == 'clarification':
            return f"[Clarification] 需要澄清您的问题：'{user_input[:30]}...' 能详细说明吗？"
        
        return f"[Default] 收到：'{user_input[:30]}...'"
    
    def on_user_followup(self, user_response: str) -> Optional[Dict]:
        """
        收到用户后续回复时调用，评估上一条交互的质量。
        
        这是反馈学习的关键入口。
        """
        if len(self.conversation_history) < 2:
            return None
        
        last_assistant_msg = self.conversation_history[-1]['content']
        
        # 评估反馈
        feedback_signal = self.evaluator.evaluate(
            last_assistant_msg=last_assistant_msg,
            next_user_msg=user_response,
            conversation_history=self.conversation_history
        )
        
        # 找到上一条用户输入并生成 embedding 用于网络更新
        last_user_input = None
        last_strategy = None
        for turn in reversed(self.conversation_history[:-1]):
            if turn.get('role') == 'user':
                last_user_input = turn['content']
                break
        
        # 从交互历史中找到对应的策略
        if last_user_input and hasattr(self, '_interaction_log') and self._interaction_log:
            last_record = self._interaction_log[-1]
            last_strategy = last_record.get('strategy_used')
        
        # 如果找到了上一条交互，更新直觉网络
        if last_user_input and last_strategy:
            query_emb = self.embedder(last_user_input)
            self.intuition.update(
                query_embedding=query_emb,
                strategy_id=last_strategy,
                reward=feedback_signal.reward
            )
        
        result = {
            'feedback_type': feedback_signal.signal_type.name,
            'reward': feedback_signal.reward,
            'confidence': feedback_signal.confidence,
            'explanation': feedback_signal.explanation
        }
        
        self.stats['total_reward'] += feedback_signal.reward
        
        # 如果是负反馈，可以触发深度搜索
        if feedback_signal.reward < -0.3:
            result['triggered_deep_search'] = True
        
        return result
    
    def update_from_feedback(
        self,
        query_embedding: List[float],
        strategy_id: str,
        reward: float
    ) -> Dict:
        """
        使用反馈更新直觉网络。
        
        这是训练直觉网络的核心方法。
        """
        stats = self.intuition.update(
            query_embedding=query_embedding,
            strategy_id=strategy_id,
            reward=reward
        )
        
        return {
            'loss': stats['policy_loss'],
            'log_prob': stats['log_prob'],
            'temperature': stats['temperature'],
            'total_updates': self.intuition.training_stats['updates']
        }
    
    def get_stats(self) -> Dict:
        """获取代理统计信息。"""
        return {
            **self.stats,
            'intuition_stats': self.intuition.training_stats,
            'evaluator_stats': self.evaluator.get_stats(),
            'avg_reward': self.stats['total_reward'] / max(1, self.stats['total_interactions'])
        }
    
    def save(self, path: str):
        """保存模型状态。"""
        self.intuition.save(path)
    
    def reset_conversation(self):
        """重置对话状态。"""
        self.conversation_history = []
        self.last_interaction_id = None
        self._interaction_log = []


def demo():
    """
    演示 MemabraAgent 的使用。
    """
    print("=" * 60)
    print("Memabra 核心集成演示")
    print("=" * 60)
    
    # 创建代理
    agent = MemabraAgent()
    
    # 模拟对话场景
    scenarios = [
        # (用户输入, 后续用户回复, 描述)
        ("今天天气怎么样？", "谢谢，我知道了", "正向反馈场景"),
        ("帮我查一下股票价格", "不对，我说的是另一只股票", "负向反馈场景"),
        ("解释一下量子计算", "还有呢？", "追问场景"),
        ("写一个Python函数", "可以了，谢谢", "成功场景"),
    ]
    
    for user_input, followup, description in scenarios:
        print(f"\n{'='*40}")
        print(f"场景: {description}")
        print(f"{'='*40}")
        
        # 处理用户输入
        result = agent.process(user_input)
        print(f"\n用户: {user_input}")
        print(f"策略: {result['strategy']} (置信度: {result['confidence']:.2f})")
        print(f"回复: {result['response']}")
        
        # 模拟用户后续回复
        print(f"\n用户后续: {followup}")
        feedback = agent.on_user_followup(followup)
        if feedback:
            print(f"反馈类型: {feedback['feedback_type']}")
            print(f"奖励值: {feedback['reward']:+.2f}")
            print(f"解释: {feedback['explanation']}")
        
        # 更新网络
        # 获取 query_embedding（简化演示）
        query_emb = agent.embedder(user_input)
        update_stats = agent.update_from_feedback(
            query_embedding=query_emb,
            strategy_id=result['strategy'],
            reward=feedback['reward'] if feedback else 0.0
        )
        print(f"网络更新: loss={update_stats['loss']:.4f}, temp={update_stats['temperature']:.3f}")
    
    # 最终统计
    print(f"\n{'='*60}")
    print("最终统计")
    print(f"{'='*60}")
    stats = agent.get_stats()
    print(f"总交互数: {stats['total_interactions']}")
    print(f"策略使用分布: {stats['strategy_usage']}")
    print(f"平均奖励: {stats['avg_reward']:+.3f}")
    print(f"直觉网络更新次数: {stats['intuition_stats']['updates']}")


if __name__ == "__main__":
    demo()
