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


class HierarchicalMemory:
    """
    Hierarchical memory system combining episodic, semantic, and procedural memory.
    """
    
    def __init__(self, embedding_fn=None):
        self.episodic = EpisodicStore(embedding_fn)
        self.semantic = SemanticStore(embedding_fn)
        self.procedural = ProceduralStore(embedding_fn)
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
            Dict with keys 'episodic', 'semantic', 'procedural'
        """
        query_emb = self.embed(query_text) if self.embed else None
        
        results = {
            'episodic': [],
            'semantic': [],
            'procedural': []
        }
        
        if strategy_id == 'direct_answer':
            # Prioritize semantic memory (facts)
            if query_emb:
                results['semantic'] = self.semantic.search(query_emb, top_k)
            results['procedural'] = self.procedural.find_matching_skills(query_text)[:top_k]
        
        elif strategy_id == 'search_required':
            # Prioritize episodic memory (past interactions)
            if query_emb:
                results['episodic'] = self.episodic.search(query_emb, top_k)
                results['semantic'] = self.semantic.search(query_emb, top_k // 2)
        
        elif strategy_id == 'tool_use':
            # Prioritize procedural memory (skills)
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
            'lesson': EpisodicMemory,
            'base': Memory,
        }
        
        store_map = {
            'episodic': self.episodic,
            'semantic': self.semantic,
            'procedural': self.procedural,
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
        if isinstance(memory, EpisodicMemory):
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
        
        return result
