# my-agent

学 Agent 的动手项目。配套课程：`../awesome-agent-engineering/agent-lessons/`（9 节课）。

**这个仓库里的代码是我自己一行行敲出来的**，课程仓库只当参考资料——所以每课一个目录，
从零手写，不抄课程成品。目标是把 L03（ReAct 循环）和 L09（毕业项目）写成
面试能讲清、能给人看的东西。

---

## 目录结构

```
my-agent/
├── .env                    ← 智谱 API Key（已配好，不进 git）
├── requirements.txt
├── README.md
└── l01_what_is_agent/      ← 第 1 课：认识 Agent
    ├── step1_llm_only.py       裸调 LLM，看它"只会说不会做"
    ├── step2_one_tool.py       一个工具 + 手写 Function Calling（无循环）
    └── step3_agent_loop.py     加上循环 = 最小 Agent
```

后面每学一课加一个目录（`l02_function_calling/`、`l03_react_loop/` …）。

---

## 环境

- Python 3.12.4
- 依赖：`zhipuai`（智谱 SDK）、`python-dotenv`
- 模型：`glm-4-flash`（免费，且支持 function calling）

```bash
pip install -r requirements.txt
```

`.env` 已从课程仓库拷过来，`ZHIPUAI_API_KEY` 是有效的。

---

## 跑法

```bash
cd "C:/Users/yuan/Desktop/万一/RAG学习/my-agent"

# ⚠️ 必须带 PYTHONIOENCODING=utf-8，否则 emoji 在 Windows 上会撞 GBK 编码报错
PYTHONIOENCODING=utf-8 python l01_what_is_agent/step1_llm_only.py
PYTHONIOENCODING=utf-8 python l01_what_is_agent/step2_one_tool.py
PYTHONIOENCODING=utf-8 python l01_what_is_agent/step3_agent_loop.py
```

`.env` 的查找方式：`load_dotenv()` 会从**脚本所在目录向上逐级找**，
所以 `l01_what_is_agent/step3_*.py` 也能找到 `my-agent/.env`。

---

## 每课学完自检

敲完一个文件，先确认没有语法/缩进错：

```bash
python -m py_compile l01_what_is_agent/step3_agent_loop.py && echo OK
```

---

## 进度

- [x] L01 认识 Agent（step1/2/3 已写完）
- [ ] L02 Function Calling 深入
- [ ] L03 ReAct 循环（面试核心）
- [ ] L04 多工具与工具设计
- [ ] L05 记忆
- [ ] L06 规划与任务分解
- [ ] L07 Agentic RAG（接 my-rag）
- [ ] L08 多智能体协作
- [ ] L09 毕业项目：智能研究助手
