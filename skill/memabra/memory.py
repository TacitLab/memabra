"""
Hierarchical Memory System: Episodic, Semantic, and Procedural memory.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid


@dataclass
class Memory:
    """Base memory class."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0  # For forgetting curve
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass  
class EpisodicMemory(Memory):
    """Memory of specific events/interactions."""
    type: str = "episodic"
    outcome: Optional[str] = None
    strategy_used: Optional[str] = None


@dataclass
class SemanticMemory(Memory):
    """Memory of facts and concepts."""
    type: str = "semantic"
    subject: str = ""
    predicate: str = ""
    object: str = ""
    source: str = ""
    confidence: float = 0.5


@dataclass
class ProceduralMemory(Memory):
    """Memory of skills and procedures."""
    type: str = "procedural"
    name: str = ""
    trigger_patterns: List[str] = field(default_factory=list)
    action: str = ""
    success_rate: float = 0.0
    avg_reward: float = 0.0
    # Extended: structured tool experience
    tool_name: Optional[str] = None
    tool_params_schema: Optional[Dict[str, Any]] = None
    total_calls: int = 0
    avg_latency_ms: float = 0.0
    context_tags: List[str] = field(default_factory=list)


@dataclass
class ActionStep:
    """A single step in an action chain."""
    step_index: int = 0
    action_type: str = ""  # "tool_call", "skill_call", "memory_retrieve", "llm_generate"
    tool_or_skill: str = ""  # tool/skill name
    params: Dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    success: bool = True
    latency_ms: float = 0.0
    timestamp: str = ""


@dataclass
class ActionMemory(Memory):
    """
    Memory of complete action chains — records structured tool/skill usage.
    
    This is the 'broad-sense' memory that captures HOW the agent solved problems,
    not just WHAT was said. Each ActionMemory records a full chain:
    user_query → step1 (tool_call) → step2 (skill_call) → ... → final_response
    """
    type: str = "action"
    user_query: str = ""
    strategy_used: str = ""
    action_chain: List[Dict[str, Any]] = field(default_factory=list)  # serialized ActionSteps
    final_response_summary: str = ""
    total_steps: int = 0
    total_latency_ms: float = 0.0
    reward: float = 0.0
    context_tags: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)  # quick lookup list
    success: bool = True


class MemoryStore:
    """Base class for memory stores."""
    
    def __init__(self, embedding_fn=None):
        self.memories: Dict[str, Memory] = {}
        self.embed = embedding_fn
    
    def add(self, memory: Memory) -> str:
        """Add a memory and return its ID."""
        self.memories[memory.id] = memory
        return memory.id
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        memory = self.memories.get(memory_id)
        if memory:
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
        return memory
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Memory]:
        """Search memories by embedding similarity."""
        # Simple linear search for MVP
        scored = []
        for mem in self.memories.values():
            if mem.embedding:
                score = self._cosine_sim(query_embedding, mem.embedding)
                scored.append((score, mem))
        
        scored.sort(reverse=True)
        return [mem for _, mem in scored[:top_k]]
    
    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        import numpy as np
        a_vec = np.array(a)
        b_vec = np.array(b)
        
        dot = np.dot(a_vec, b_vec)
        norm_a = np.linalg.norm(a_vec)
        norm_b = np.linalg.norm(b_vec)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot / (norm_a * norm_b))
    
    def apply_forgetting(self, current_time: Optional[datetime] = None):
        """Apply forgetting curve to all memories."""
        if current_time is None:
            current_time = datetime.utcnow()
        
        for mem in self.memories.values():
            age = (current_time - mem.timestamp).total_seconds()
            # Ebbinghaus forgetting curve: R = e^(-t/S)
            # where S is stability, modified by access count
            stability = 86400 * 5 * (1 + 0.1 * mem.access_count)  # 5 days base
            retention = mem.strength * (2.718 ** (-age / stability))
            mem.strength = retention


class EpisodicStore(MemoryStore):
    """Store for episodic memories."""
    
    def add_interaction(self, input_text: str, output_text: str, 
                        strategy: str, outcome: Optional[str] = None) -> str:
        """Record an interaction."""
        embedding = self.embed(input_text) if self.embed else None
        
        memory = EpisodicMemory(
            content=f"Input: {input_text}\nOutput: {output_text}",
            embedding=embedding,
            outcome=outcome,
            strategy_used=strategy,
            metadata={
                'input': input_text,
                'output': output_text,
            }
        )
        return self.add(memory)
    
    def get_recent_context(self, n: int = 5) -> List[EpisodicMemory]:
        """Get recent interactions."""
        sorted_memories = sorted(
            [m for m in self.memories.values() if isinstance(m, EpisodicMemory)],
            key=lambda m: m.timestamp,
            reverse=True
        )
        return sorted_memories[:n]


class SemanticStore(MemoryStore):
    """Store for semantic memories."""
    
    def add_fact(self, subject: str, predicate: str, obj: str, 
                 source: str = "", confidence: float = 0.5) -> str:
        """Add a factual triple."""
        content = f"{subject} {predicate} {obj}"
        embedding = self.embed(content) if self.embed else None
        
        memory = SemanticMemory(
            content=content,
            embedding=embedding,
            subject=subject,
            predicate=predicate,
            object=obj,
            source=source,
            confidence=confidence
        )
        return self.add(memory)


class ProceduralStore(MemoryStore):
    """Store for procedural memories."""
    
    def add_skill(self, name: str, trigger_patterns: List[str], 
                  action: str) -> str:
        """Add a skill/procedure."""
        content = f"Skill: {name}\nAction: {action}"
        
        memory = ProceduralMemory(
            content=content,
            name=name,
            trigger_patterns=trigger_patterns,
            action=action
        )
        return self.add(memory)
    
    def find_matching_skills(self, input_text: str) -> List[ProceduralMemory]:
        """Find skills matching input patterns."""
        matches = []
        for mem in self.memories.values():
            if isinstance(mem, ProceduralMemory):
                for pattern in mem.trigger_patterns:
                    if pattern in input_text:
                        matches.append(mem)
                        break
        return matches


class ActionStore(MemoryStore):
    """Store for action chain memories — records structured tool/skill usage."""
    
    def record_action_chain(
        self,
        user_query: str,
        strategy_used: str,
        action_chain: List[Dict[str, Any]],
        final_response_summary: str,
        reward: float = 0.0,
        context_tags: Optional[List[str]] = None,
        success: bool = True,
    ) -> str:
        """
        Record a complete action chain from a single interaction.
        
        Args:
            user_query: The original user input
            strategy_used: Strategy selected by intuition network
            action_chain: List of ActionStep dicts, each containing:
                - step_index, action_type, tool_or_skill, params,
                  result_summary, success, latency_ms, timestamp
            final_response_summary: Summary of the final response
            reward: Feedback reward for this chain
            context_tags: Tags for retrieval (e.g., ["file_search", "python"])
            success: Whether the overall chain succeeded
        """
        tools_used = list({
            step.get('tool_or_skill', '')
            for step in action_chain
            if step.get('tool_or_skill')
        })
        total_latency = sum(step.get('latency_ms', 0) for step in action_chain)
        
        content = f"Query: {user_query}\nTools: {', '.join(tools_used)}\nResult: {final_response_summary}"
        embedding = self.embed(content) if self.embed else None
        
        memory = ActionMemory(
            content=content,
            embedding=embedding,
            user_query=user_query,
            strategy_used=strategy_used,
            action_chain=action_chain,
            final_response_summary=final_response_summary,
            total_steps=len(action_chain),
            total_latency_ms=total_latency,
            reward=reward,
            context_tags=context_tags or [],
            tools_used=tools_used,
            success=success,
        )
        return self.add(memory)
    
    def find_similar_chains(self, query_text: str, top_k: int = 5) -> List[ActionMemory]:
        """Find action chains similar to the given query using embedding search."""
        if not self.embed:
            return []
        query_emb = self.embed(query_text)
        results = self.search(query_emb, top_k)
        return [m for m in results if isinstance(m, ActionMemory)]
    
    def find_by_tool(self, tool_name: str, top_k: int = 10) -> List[ActionMemory]:
        """Find action chains that used a specific tool."""
        matches = []
        for mem in self.memories.values():
            if isinstance(mem, ActionMemory) and tool_name in mem.tools_used:
                matches.append(mem)
        matches.sort(key=lambda m: m.timestamp, reverse=True)
        return matches[:top_k]
    
    def find_successful_patterns(self, tool_name: str, min_reward: float = 0.5) -> List[ActionMemory]:
        """Find high-reward action chains for a tool — learn what works."""
        return [
            m for m in self.memories.values()
            if isinstance(m, ActionMemory)
            and tool_name in m.tools_used
            and m.reward >= min_reward
            and m.success
        ]
    
    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Aggregate statistics per tool across all action memories."""
        stats: Dict[str, Dict[str, Any]] = {}
        for mem in self.memories.values():
            if not isinstance(mem, ActionMemory):
                continue
            for step in mem.action_chain:
                tool = step.get('tool_or_skill', '')
                if not tool:
                    continue
                if tool not in stats:
                    stats[tool] = {
                        'total_calls': 0,
                        'successes': 0,
                        'total_latency_ms': 0.0,
                        'total_reward': 0.0,
                    }
                stats[tool]['total_calls'] += 1
                if step.get('success', True):
                    stats[tool]['successes'] += 1
                stats[tool]['total_latency_ms'] += step.get('latency_ms', 0)
                stats[tool]['total_reward'] += mem.reward
        
        # Compute averages
        for tool, s in stats.items():
            n = s['total_calls']
            s['success_rate'] = s['successes'] / n if n else 0.0
            s['avg_latency_ms'] = s['total_latency_ms'] / n if n else 0.0
            s['avg_reward'] = s['total_reward'] / n if n else 0.0
        
        return stats


class HierarchicalMemory:
    """
    Hierarchical memory system combining episodic, semantic, procedural, and action memory.
    """
    
    def __init__(self, embedding_fn=None):
        self.episodic = EpisodicStore(embedding_fn)
        self.semantic = SemanticStore(embedding_fn)
        self.procedural = ProceduralStore(embedding_fn)
        self.action = ActionStore(embedding_fn)
        self.embed = embedding_fn
    
    def retrieve(self, query_text: str, strategy_id: str, 
                 top_k: int = 5) -> Dict[str, List[Memory]]:
        """
        Retrieve memories based on strategy.
        
        Args:
            query_text: The query text
            strategy_id: Strategy to use for retrieval
            top_k: Number of memories to retrieve per type
            
        Returns:
            Dict with keys 'episodic', 'semantic', 'procedural', 'action'
        """
        query_emb = self.embed(query_text) if self.embed else None
        
        results = {
            'episodic': [],
            'semantic': [],
            'procedural': [],
            'action': [],
        }
        
        if strategy_id == 'direct_answer':
            # Prioritize semantic memory (facts)
            if query_emb:
                results['semantic'] = self.semantic.search(query_emb, top_k)
            results['procedural'] = self.procedural.find_matching_skills(query_text)[:top_k]
            # Also check action history for similar queries
            results['action'] = self.action.find_similar_chains(query_text, top_k // 2)
        
        elif strategy_id == 'search_required':
            # Prioritize episodic memory (past interactions)
            if query_emb:
                results['episodic'] = self.episodic.search(query_emb, top_k)
                results['semantic'] = self.semantic.search(query_emb, top_k // 2)
            results['action'] = self.action.find_similar_chains(query_text, top_k // 2)
        
        elif strategy_id == 'tool_use':
            # Prioritize action memory (past tool usage) + procedural memory
            results['action'] = self.action.find_similar_chains(query_text, top_k)
            results['procedural'] = self.procedural.find_matching_skills(query_text)
            if query_emb:
                results['episodic'] = self.episodic.search(query_emb, top_k // 2)
        
        elif strategy_id == 'clarification':
            # Get recent context
            results['episodic'] = self.episodic.get_recent_context(top_k)
        
        return results
    
    def store_lesson(self, problem: str, failed_strategy: str, 
                     user_feedback: str, better_approach: str):
        """Store a lesson learned from failure."""
        lesson_content = f"Problem: {problem}\nFailed: {failed_strategy}\nFeedback: {user_feedback}\nBetter: {better_approach}"
        
        embedding = self.embed(lesson_content) if self.embed else None
        
        lesson = EpisodicMemory(
            content=lesson_content,
            embedding=embedding,
            type="lesson",
            metadata={
                'problem': problem,
                'failed_strategy': failed_strategy,
                'user_feedback': user_feedback,
                'better_approach': better_approach
            }
        )
        self.episodic.add(lesson)
    
    def save_to_disk(self, path: str):
        """Save all memories to disk."""
        data = {
            'episodic': [self._memory_to_dict(m) for m in self.episodic.memories.values()],
            'semantic': [self._memory_to_dict(m) for m in self.semantic.memories.values()],
            'procedural': [self._memory_to_dict(m) for m in self.procedural.memories.values()],
            'action': [self._memory_to_dict(m) for m in self.action.memories.values()],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_from_disk(self, path: str):
        """Load memories from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Type reconstruction map
        type_map = {
            'episodic': EpisodicMemory,
            'semantic': SemanticMemory,
            'procedural': ProceduralMemory,
            'action': ActionMemory,
            'lesson': EpisodicMemory,
            'base': Memory,
        }
        
        store_map = {
            'episodic': self.episodic,
            'semantic': self.semantic,
            'procedural': self.procedural,
            'action': self.action,
        }
        
        for store_key, entries in data.items():
            store = store_map.get(store_key)
            if store is None:
                continue
            for entry in entries:
                mem_type = entry.get('type', 'base')
                cls = type_map.get(mem_type, Memory)
                
                # Build base fields
                kwargs = {
                    'id': entry.get('id', str(uuid.uuid4())),
                    'content': entry.get('content', ''),
                    'embedding': entry.get('embedding'),
                    'timestamp': datetime.fromisoformat(entry['timestamp']) if 'timestamp' in entry else datetime.utcnow(),
                    'metadata': entry.get('metadata', {}),
                    'strength': entry.get('strength', 1.0),
                    'access_count': entry.get('access_count', 0),
                }
                
                # Add type-specific fields
                if cls == EpisodicMemory:
                    kwargs['outcome'] = entry.get('metadata', {}).get('outcome')
                    kwargs['strategy_used'] = entry.get('metadata', {}).get('strategy_used')
                elif cls == SemanticMemory:
                    kwargs['subject'] = entry.get('metadata', {}).get('subject', '')
                    kwargs['predicate'] = entry.get('metadata', {}).get('predicate', '')
                    kwargs['object'] = entry.get('metadata', {}).get('object', '')
                    kwargs['source'] = entry.get('metadata', {}).get('source', '')
                    kwargs['confidence'] = entry.get('metadata', {}).get('confidence', 0.5)
                elif cls == ProceduralMemory:
                    kwargs['name'] = entry.get('metadata', {}).get('name', '')
                    kwargs['trigger_patterns'] = entry.get('metadata', {}).get('trigger_patterns', [])
                    kwargs['action'] = entry.get('metadata', {}).get('action', '')
                    kwargs['success_rate'] = entry.get('metadata', {}).get('success_rate', 0.0)
                    kwargs['avg_reward'] = entry.get('metadata', {}).get('avg_reward', 0.0)
                    kwargs['tool_name'] = entry.get('metadata', {}).get('tool_name')
                    kwargs['tool_params_schema'] = entry.get('metadata', {}).get('tool_params_schema')
                    kwargs['total_calls'] = entry.get('metadata', {}).get('total_calls', 0)
                    kwargs['avg_latency_ms'] = entry.get('metadata', {}).get('avg_latency_ms', 0.0)
                    kwargs['context_tags'] = entry.get('metadata', {}).get('context_tags', [])
                elif cls == ActionMemory:
                    meta = entry.get('metadata', {})
                    kwargs['user_query'] = meta.get('user_query', '')
                    kwargs['strategy_used'] = meta.get('strategy_used', '')
                    kwargs['action_chain'] = meta.get('action_chain', [])
                    kwargs['final_response_summary'] = meta.get('final_response_summary', '')
                    kwargs['total_steps'] = meta.get('total_steps', 0)
                    kwargs['total_latency_ms'] = meta.get('total_latency_ms', 0.0)
                    kwargs['reward'] = meta.get('reward', 0.0)
                    kwargs['context_tags'] = meta.get('context_tags', [])
                    kwargs['tools_used'] = meta.get('tools_used', [])
                    kwargs['success'] = meta.get('success', True)
                
                memory = cls(**kwargs)
                store.add(memory)
    
    def _memory_to_dict(self, memory: Memory) -> Dict:
        """Convert memory to dictionary."""
        result = {
            'id': memory.id,
            'type': getattr(memory, 'type', 'base'),
            'content': memory.content,
            'embedding': memory.embedding,
            'timestamp': memory.timestamp.isoformat(),
            'metadata': dict(memory.metadata),
            'strength': memory.strength,
            'access_count': memory.access_count,
        }
        
        # Persist type-specific fields inside metadata for reconstruction
        if isinstance(memory, ActionMemory):
            # Check ActionMemory before EpisodicMemory since it's not a subclass
            result['metadata']['user_query'] = memory.user_query
            result['metadata']['strategy_used'] = memory.strategy_used
            result['metadata']['action_chain'] = memory.action_chain
            result['metadata']['final_response_summary'] = memory.final_response_summary
            result['metadata']['total_steps'] = memory.total_steps
            result['metadata']['total_latency_ms'] = memory.total_latency_ms
            result['metadata']['reward'] = memory.reward
            result['metadata']['context_tags'] = memory.context_tags
            result['metadata']['tools_used'] = memory.tools_used
            result['metadata']['success'] = memory.success
        elif isinstance(memory, EpisodicMemory):
            result['metadata']['outcome'] = memory.outcome
            result['metadata']['strategy_used'] = memory.strategy_used
        elif isinstance(memory, SemanticMemory):
            result['metadata']['subject'] = memory.subject
            result['metadata']['predicate'] = memory.predicate
            result['metadata']['object'] = memory.object
            result['metadata']['source'] = memory.source
            result['metadata']['confidence'] = memory.confidence
        elif isinstance(memory, ProceduralMemory):
            result['metadata']['name'] = memory.name
            result['metadata']['trigger_patterns'] = memory.trigger_patterns
            result['metadata']['action'] = memory.action
            result['metadata']['success_rate'] = memory.success_rate
            result['metadata']['avg_reward'] = memory.avg_reward
            result['metadata']['tool_name'] = memory.tool_name
            result['metadata']['tool_params_schema'] = memory.tool_params_schema
            result['metadata']['total_calls'] = memory.total_calls
            result['metadata']['avg_latency_ms'] = memory.avg_latency_ms
            result['metadata']['context_tags'] = memory.context_tags
        
        return result
