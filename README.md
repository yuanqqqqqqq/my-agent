# my-agent

一个功能完整的 **AI 研究助手（AI Research Agent）**。

给定一个研究主题，它能自动完成：**规划要点 → 多源检索（联网 + 本地知识库）→ 生成带来源的结构化报告 → 代码校验 → 不合格自动打回重做**。

基于智谱 GLM-4-Flash 构建，纯 Python 实现，支持 CLI / 交互 / Markdown / JSON 多种使用方式。

> 本项目是在学习 Agent 技术的过程中、借助 AI 辅助实现的。核心特色是用**确定性的代码校验层**替代 LLM 审查者，从机制上抑制模型「编造来源 / 编数字」。

[![CI](https://github.com/yuanqqqqqqq/my-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/yuanqqqqqqq/my-agent/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 功能特性

### 🔍 多源检索
- **联网搜索**：DuckDuckGo 搜索，内置相关性过滤，自动丢弃「成功但无关」的结果
- **网页正文抓取**：对搜索结果进一步抓取全文，突破摘要限制
- **本地知识库（RAG）**：复用自建的 my-rag，检索本地文档（如 FastAPI 官方文档）
- **精确计算**：四则运算、取余、幂等

### 🧠 自主工作流
- **自动规划**：接到主题后先把研究拆解成 3~5 个要点
- **按需检索**：模型自己决定查什么、查几次，而不是无脑全查
- **结构化报告**：概述 + 要点（每条附来源）+ 来源清单

### 🛡️ 确定性校验层（核心特色）
不调用任何模型，纯代码校验，从机制上拦住幻觉：

| 校验项 | 作用 |
|--------|------|
| 引用真实性 | 报告里的来源必须是本次真实检索到的 |
| 数字可溯源 | 报告里的每个数字都要能在工具返回原文里找到 |
| 要点覆盖 | 计划里的要点，报告必须全部覆盖 |
| 空壳检测 | 不能只有概述、没有实质要点 |
| 私有限定词 | 问「我们公司 / 我」这类私有信息时，不能用公开资料冒充 |

不合格自动打回重做（最多 2 次），直到通过、或如实承认资料不足。**「资料不足、老实拒答」是合格产出，不会被误拦**。

### 💬 多轮对话
交互模式下连续追问，自动携带前面几轮的上下文，并用滑动窗口截断防止上下文膨胀。

### 🖥️ 多种使用方式
- 命令行：`my-agent "主题"`
- 交互模式：`my-agent -i`
- 导出 Markdown：`--out report.md`
- 纯 JSON 输出：`--json`（可被程序消费）

---

## 快速开始

- Python 3.12

```bash
pip install -r requirements.txt      # 或：pip install -e .（附带 my-agent 命令行）
cp .env.example .env                 # 填入智谱 API Key（https://bigmodel.cn/）
```

> ⚠️ Windows 下请带 `PYTHONIOENCODING=utf-8`，否则 emoji 会撞 GBK 编码报错。

### 使用示例

```bash
# 终端看报告
PYTHONIOENCODING=utf-8 python research_agent/agent.py "FastAPI 是怎么处理文件上传的？"

# 导出 Markdown / 纯 JSON
PYTHONIOENCODING=utf-8 python research_agent/agent.py "主题" --out report.md
PYTHONIOENCODING=utf-8 python research_agent/agent.py "主题" --json

# 交互模式（连续追问带上下文）
PYTHONIOENCODING=utf-8 python research_agent/agent.py -i
```

> 需要本地知识库检索（`search_kb`）时，先安装 my-rag：`cd ../my-rag && pip install -e .`。
> 没装它也能用联网研究等其余全部功能，只是 `search_kb` 会提示先安装。

### 运行测试

```bash
python tests/test_verify.py     # 校验层单元测试（纯函数，不联网、不要 Key）
python research_agent/verify.py # 校验层内置自检
python research_agent/tools.py  # 工具自检
```

---

## 演示

### 研究助手：需要联网的题目

本地知识库只装了 FastAPI 文档，问它「大模型幻觉」这类题时，它会自主转向联网，并在校验层两次打回后写出实质内容：

```bash
PYTHONIOENCODING=utf-8 python research_agent/agent.py "大模型的幻觉问题有哪些主流缓解方法？"
```

```
📋 研究计划（1 条）：大模型的幻觉问题有哪些主流的缓解方法

  📚 本地知识库 search_kb → 命中 3 条（都是 FastAPI 文档，与题目无关）
❌ 校验不通过：① 引用了不存在的来源「没有找到相关资料」 ④ 所有要点都太短

  🌐 联网搜索 search_web → 搜到 5 条相关结果
❌ 校验不通过：④ 所有要点都太短

  🌐 联网搜索 search_web → 搜到 5 条相关结果
✅ 校验通过（引用真实性 / 数字出处 / 要点覆盖 / 非空壳 / 尽力程度）

📄 研究报告：概述给出 6 种缓解方法——数据源优化、训练过程改进、推理策略调整、
          模型结构优化、后验幻觉检测、知识增强（RAG），来源全部来自真实检索到的网页。
```

### ReAct 推理链（l03）

纯文本 ReAct 循环，每一步的思考与动作都可见：

```
问题：现在几点？把小时和分钟相加，再除以 7 求余数

第 1 轮  Thought: 先获取当前时间
         Action: get_current_time()          → 2026-09-29 10:13:10
第 2 轮  Thought: 10 点 13 分，先把小时和分钟相加
         Action: calculator('10 + 13')       → 23
第 3 轮  Thought: 再用 23 除以 7 求余数
         Action: calculator('23 % 7')        → 2
第 4 轮  Final Answer: 余数是 2 ✅
```

---

## 项目结构

```
my-agent/
├── research_agent/       ★ 研究助手本体
│   ├── agent.py          主循环：规划 → 检索 → 报告 → 校验 → 打回重做
│   ├── tools.py          联网搜索 / 抓正文 / 查知识库 / 计算，每件工具自带有效性校验
│   ├── verify.py         确定性校验层（引用 / 数字 / 覆盖 / 空壳 / 私有限定词）
│   └── eval.py           能力评测（有校验层 vs 无校验层）
├── tests/test_verify.py  校验层单元测试
├── l01_what_is_agent/    实现过程沉淀的各模块（从最小 Agent 到多智能体）
├── l02_function_calling/
├── l03_react_loop/
├── l04_tool_design/
├── l05_memory/
├── l06_planning/
├── l07_agentic_rag/
├── l08_multi_agent/
├── pyproject.toml        打包 + my-agent 命令行入口
└── .github/workflows/    CI（语法 + 测试，不依赖 API Key）
```

---

## 技术设计

- **结构化输出走 function calling**（`submit_plan` / `submit_report`），而不是 prompt 里要求「输出 JSON」——后者容易被模型当成任务步骤写进计划
- **校验用确定性代码，不用 LLM**——LLM 当审查者容易变成橡皮图章，代码校验可复现、零 token
- **工具自带有效性校验**：联网搜索按关键词命中率过滤「成功但无关」的结果
- **正确拒答不是失败**：知识库里没有就如实说没有，而不是硬编一个

---

## License

[MIT](LICENSE) © 2026 yuanqqqqqqq
