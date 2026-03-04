# 开发路线图

## Phase 1: MVP (2-3 周)

**目标**：验证核心循环是否 work

### 任务清单

- [ ] **项目骨架**
  - [ ] 目录结构搭建
  - [ ] 基础依赖配置 (requirements.txt, pyproject.toml)
  - [ ] 配置管理系统

- [ ] **简化版直觉网络**
  - [ ] 基于余弦相似度的策略选择
  - [ ] 4个基础策略: direct_answer, search_required, tool_use, clarification
  - [ ] 在线更新机制 (简单梯度下降)

- [ ] **基础记忆系统**
  - [ ] ChromaDB 向量存储集成
  - [ ] 单一记忆类型 (Episodic)
  - [ ] 简单的嵌入生成 (sentence-transformers)

- [ ] **隐式反馈 v1**
  - [ ] 关键词检测 (谢谢/错了)
  - [ ] 语义相似度判断 (重复追问)
  - [ ] 基础奖励分配

- [ ] **端到端测试**
  - [ ] 模拟对话测试
  - [ ] 观察直觉网络是否随反馈调整

### 验收标准
- 能运行一个完整对话循环
- 负反馈后能看到策略权重变化

---

## Phase 2: 核心完善 (3-4 周)

**目标**：完善记忆分层和反馈系统

### 任务清单

- [ ] **三层记忆系统**
  - [ ] Episodic: 对话历史 + 时间索引
  - [ ] Semantic: 知识图谱 (NetworkX)
  - [ ] Procedural: 技能库 + 触发模式

- [ ] **隐式反馈 v2**
  - [ ] 意图识别模型
  - [ ] 延迟奖励分配 (n-step return)
  - [ ] 对话成功率判断

- [ ] **探索模式**
  - [ ] 低置信度时的多策略尝试
  - [ ] 结果比较和选择机制
  - [ ] "教训"记录和检索

- [ ] **神经网络直觉网络**
  - [ ] 小型 MLP 实现
  - [ ] REINFORCE 训练
  - [ ] 与简化版对比评估

- [ ] **记忆巩固与遗忘**
  - [ ] 艾宾浩斯遗忘曲线实现
  - [ ] 睡眠/空闲时的记忆重放
  - [ ] 重要记忆标记和保护

### 验收标准
- 三层记忆能协同工作
- 能处理多轮复杂对话
- 直觉网络有一定"学习"表现

---

## Phase 3: 高级特性 (4-6 周)

**目标**：生产可用级别的稳定性和性能

### 任务清单

- [ ] **多 Agent 支持**
  - [ ] Agent 间记忆共享接口
  - [ ] 社会记忆 (其他 Agent 的经验)
  - [ ] 记忆权限和隐私控制

- [ ] **可观测性**
  - [ ] 记忆检索可视化
  - [ ] 直觉决策追踪
  - [ ] 奖励流 dashboard

- [ ] **性能优化**
  - [ ] 向量检索优化 (HNSW 索引)
  - [ ] 批处理和缓存
  - [ ] 增量保存和恢复

- [ ] **OpenClaw 集成**
  - [ ] 作为 Skill 封装
  - [ ] 与现有 memory 系统兼容
  - [ ] 配置文件支持

- [ ] **评估基准**
  - [ ] 对话成功率 benchmark
  - [ ] 记忆检索准确度测试
  - [ ] 学习效率对比实验

### 验收标准
- 可集成到 OpenClaw
- 有完整的评估数据
- 文档齐全

---

## Phase 4: 前沿探索 (长期)

**目标**：探索更前沿的方向

### 可能方向

- [ ] **多模态记忆**
  - 图像、音频的记忆存储和检索
  - 跨模态关联

- [ ] **元学习 (Meta-Learning)**
  - 快速适应新用户
  - 跨任务迁移

- [ ] **生物启发机制**
  - 海马体重放 (Hippocampal Replay)
  - 记忆巩固的睡眠模拟
  - 情绪标签和记忆优先级

- [ ] **可解释性**
  - 直觉决策的可视化解释
  - 记忆溯源 (为什么调用这段记忆)

---

## 技术债务追踪

| 问题 | 引入阶段 | 计划解决阶段 | 优先级 |
|------|----------|--------------|--------|
| 单线程执行 | Phase 1 | Phase 2 | 高 |
| 内存存储无持久化 | Phase 1 | Phase 1 | 高 |
| 硬编码阈值 | Phase 1 | Phase 2 | 中 |
| 无并发控制 | Phase 2 | Phase 3 | 中 |

---

## 贡献指南

### 如何参与

1. **从 Good First Issue 开始**
   - 文档改进
   - 测试用例补充
   - 配置示例

2. **功能开发流程**
   ```
   1. 在 Discussion 中提出设计
   2. 维护者确认方向
   3. Fork → Branch → PR
   4. Code Review
   5. Merge
   ```

3. **代码规范**
   - Python: Black + isort + flake8
   - 类型注解: mypy
   - 测试: pytest
   - 文档: Google Style Docstrings

---

## 里程碑时间线

```
Week 1-3:   [====] Phase 1 MVP
Week 4-7:   [====] Phase 2 核心完善
Week 8-13:  [====] Phase 3 高级特性
Week 14+:   [====] Phase 4 前沿探索 (持续)
```

---

## 参考资源

### 论文
- "AI Meets Brain: A Unified Survey on Memory Systems from Cognitive Neuroscience to Autonomous Agents" (2025)
- "Memory in the Age of AI Agents" (2026)
- "A-Mem: Agentic Memory for LLM Agents" (2025)

### 开源项目
- MemGPT
- LangChain Memory
- ChromaDB
- NetworkX

### 工具
- sentence-transformers (嵌入)
- ChromaDB (向量存储)
- PyTorch (神经网络)
- pytest (测试)
