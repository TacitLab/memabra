# Memabra

**Memabra** = **Mem**ory + Cere**bra** (Brain Cortex)

A bio-inspired memory and intuition system for AI agents.

<details>
<summary>🌏 切换到中文 / Switch to Chinese</summary>

## Memabra 简介

**Memabra** = **Mem**ory（记忆）+ Cere**bra**（大脑皮层）

一个受生物记忆机制启发的 AI Agent 记忆与直觉系统。

### 核心理念

人类解决问题时，首先依赖**直觉**快速判断该用什么方式处理，然后调用相关记忆，最后执行。如果结果有效，这个路径会被强化；如果失败，则触发深度搜索和模式修正。

Memabra 尝试将这种双系统思维（直觉 + 分析）引入 AI Agent 的记忆管理。

### 设计原则

1. **生物启发**：借鉴人脑的记忆形成、巩固、遗忘机制
2. **隐式反馈**：不依赖用户显式评价，从交互流中自然读取信号
3. **端到端学习**：直觉网络随经验自我优化
4. **渐进复杂**：从简单原型开始，逐步演化

### 快速开始

```bash
pip install -e .
```

</details>

---

## Core Philosophy

When humans encounter a problem, they first rely on **intuition** to quickly judge how to approach it, then retrieve relevant memories, and finally execute. If the result is effective, this path is reinforced; if it fails, deep search and pattern correction are triggered.

Memabra brings this dual-system thinking (intuition + analysis) into AI agent memory management.

## Design Principles

1. **Bio-inspired**: Drawing from human memory formation, consolidation, and forgetting mechanisms
2. **Implicit Feedback**: No explicit user ratings needed—signals are read naturally from interaction flow
3. **End-to-End Learning**: The intuition network self-optimizes through experience
4. **Progressive Complexity**: Start with simple prototypes, evolve gradually

## Documentation

- [Architecture](docs/architecture.md) - System architecture overview
- [Core Mechanisms](docs/core-mechanisms.md) - Intuition network, memory retrieval, execution loop
- [Feedback System](docs/feedback-system.md) - Inferring satisfaction from user behavior
- [Roadmap](docs/roadmap.md) - Implementation plan

## Quick Start

```bash
pip install -e .
```

## License

MIT
