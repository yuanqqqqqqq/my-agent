"""
L07 · Agentic RAG —— 把 my-rag 变成 Agent 的一件工具
=====================================================
这一课的主角是你自己写的 `my-rag`（已 pip install -e 装好，import rag 即可用）。

    ★ 传统 RAG：问什么都先检索（无脑、固定流水线）
    ★ Agentic RAG：把检索包装成一个工具，让模型自己决定要不要查、查几次

本文件做三件事：
  ① 把 my-rag 的 retrieve() 包装成 `search_knowledge_base` 工具
  ② 同一组问题，跑「Agentic RAG」和「传统 RAG」对照，比检索次数和答案
  ③ 演示 Agentic RAG 的新风险：它有了「不检索」的自由，也就有了「绕过知识库自己编」的自由

跑法：PYTHONIOENCODING=utf-8 python l07_agentic_rag/agentic_rag.py
"""

import json
import os
import sys
from datetime import datetime

from zhipuai import ZhipuAI

from rag.client import get_client
from rag.retrieve import retrieve
from rag.store import get_collection

MODEL = "glm-4-flash"
INDEX_NAME = "fastapi_docs"          # 你的 my-rag 里已经建好的索引（510 块）

client = get_client()
collection = get_collection(INDEX_NAME)


# ════════════════════════════════════════════════════════════
# ① 把 my-rag 包装成工具
# ════════════════════════════════════════════════════════════
# ★ 关键设计 1：直接用 retrieve()，【不要用 ask()】。
#     ask() = 检索 + 生成。把一个「会生成答案的东西」再喂给「会生成答案的模型」，
#     你会得到「模型的模型的答案」——中间那层生成会污染信息。
#     工具只应该做【确定性】的那部分：检索。
#
# ★ 关键设计 2：返回值带【来源】。
#    这是后面「来源守卫」和 L09 毕业项目「可信度」的基础。


def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """在 FastAPI 官方文档库里检索，返回带来源的片段。"""
    hits = retrieve(client, collection, query, strategy="rerank", top_k=top_k)
    if not hits:
        return "没有检索到相关内容。知识库里只有 FastAPI 官方文档。"
    blocks = []
    for h in hits:
        blocks.append(f"[来源: {h.get('source', '?')}]\n{h.get('text', '')[:600]}")
    return "\n\n---\n\n".join(blocks)


def calculate(expression: str) -> str:
    """计算数学表达式。支持 + - * / % 和括号。"""
    allowed = set("0123456789+-*/%.() ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含不支持的字符"
    try:
        return str(eval(expression))
    except ZeroDivisionError:
        return "错误：除数不能为 0"
    except Exception as e:
        return f"计算错误：{e}"


def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOL_REGISTRY = {
    "search_knowledge_base": search_knowledge_base,
    "calculate": calculate,
    "get_current_time": get_current_time,
}

# ★ description 用 L04 的写法：做什么 + 什么时候用 + 【什么时候不用】+ 参数怎么填
TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "search_knowledge_base",
        "description": (
            "在 FastAPI 官方文档库中检索资料。"
            "当用户问 FastAPI 的用法、接口、配置、报错等具体技术问题时使用。"
            "注意：闲聊、数学计算、实时时间、以及 FastAPI 之外的技术问题，不要用这个工具。"
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string",
                      "description": "检索关键词。请提取问题里的核心术语，不要照抄整句话。"},
            "top_k": {"type": "integer", "description": "返回几个片段，默认 3"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "计算数学表达式。支持四则运算和 % 取余。需要精确计算时使用。",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "数学表达式，如 '1+1'、'17%5'"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "获取当前的日期和时间。用户问'现在几点''今天几号'时使用。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

SYSTEM_PROMPT = (
    "你是助手，可以使用工具。\n"
    "重要：凡是涉及 FastAPI 官方文档内容的问题，必须先调用 search_knowledge_base 拿材料，"
    "并且【只根据材料回答】，材料里没有就说不知道，绝对不要用你自己的记忆回答。"
)


# ════════════════════════════════════════════════════════════
# ② 两种模式
# ════════════════════════════════════════════════════════════
def run_agentic(question: str, guard: bool = False):
    """Agentic RAG：模型自己决定要不要检索。

    guard=True 时启用【来源守卫】：如果模型给出了答案但整段对话里
    一次检索都没发生，就把答案打回，强制它先查再答。
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    tools_used, retrievals = [], 0

    for _ in range(6):
        msg = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SPEC, tool_choice="auto"
        ).choices[0].message

        if msg.tool_calls:
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                tools_used.append(tc.function.name)
                if tc.function.name == "search_knowledge_base":
                    retrievals += 1
                result = TOOL_REGISTRY[tc.function.name](**args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue

        # ── 模型给出了不带工具的答案 ──
        answer = msg.content or ""

        # ★ 来源守卫：答了但一次都没检索 → 打回，明确要求它去查
        if guard and retrievals == 0:
            messages.append({"role": "assistant", "content": answer})
            messages.append({"role": "user", "content": (
                "你刚才的回答没有查知识库。请先调用 search_knowledge_base 检索，"
                "然后只根据检索到的材料回答；材料里没有就说不知道。"
            )})
            retrievals = -1          # 标记：守卫触发过一次
            continue

        return retrievals, tools_used, answer

    return retrievals, tools_used, "（撞上限）"


def run_traditional(question: str):
    """传统 RAG：不管问什么都先检索一次，再生成。"""
    retrieved = search_knowledge_base(question)
    prompt = (f"根据以下材料回答问题。材料里没有的信息，直接说不知道，不要编造。\n\n"
              f"材料：\n{retrieved}\n\n问题：{question}")
    ans = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content
    return 1, ["search_knowledge_base"], (ans or "")


# ════════════════════════════════════════════════════════════
# ③ 对照
# ════════════════════════════════════════════════════════════
QUESTIONS = [
    ("FastAPI 里怎么上传文件？", "需检索"),
    ("1+1 等于几？", "不需检索"),
    ("你好，你是谁？", "不需检索"),
    ("FastAPI 的安装命令是什么？", "需检索"),
    ("C 语言里怎么手动释放内存？", "库里没有"),
    ("现在几点了？", "不需检索（需时间工具）"),
]


def compare():
    print("=" * 74)
    print(f"Agentic RAG vs 传统 RAG   （真实索引：{INDEX_NAME}，{collection.count()} 个块）")
    print("=" * 74)

    total_a = total_t = 0
    detail = []
    for q, kind in QUESTIONS:
        ra, ta, ans_a = run_agentic(q)
        rt, tt, ans_t = run_traditional(q)
        total_a += ra
        total_t += rt
        detail.append((q, kind, ra, ta, ans_a, rt, ans_t))

    print(f"\n{'问题':<26}{'类型':<20}{'Agentic检索':<12}{'传统检索'}")
    print("-" * 74)
    for q, kind, ra, ta, ans_a, rt, ans_t in detail:
        print(f"{q[:24]:<26}{kind[:18]:<20}{ra:<12}{rt}")
    print("-" * 74)
    print(f"检索总次数：  Agentic {total_a}   传统 {total_t}   （省了 {total_t - total_a} 次）")

    print("\n" + "=" * 74)
    print("逐题看答案")
    print("=" * 74)
    for q, kind, ra, ta, ans_a, rt, ans_t in detail:
        print("─" * 74)
        print(f"❓ {q}   （{kind}）")
        print(f"  🤖 Agentic [检索 {ra} 次] 工具={ta}")
        print(f"     {(ans_a or '')[:130]}")
        print(f"  📚 传统RAG [固定检索 {rt} 次]")
        print(f"     {(ans_t or '')[:130]}")


def demo_guard():
    """演示 Agentic RAG 的新风险，以及来源守卫能不能治它。"""
    print("\n" + "=" * 74)
    print("🔬 来源守卫实验：Agentic RAG 会「绕过知识库自己答」吗？")
    print("=" * 74)
    q = "C 语言里怎么手动释放内存？"
    print(f"问题：{q}")
    print("（这题库里没有答案。理想行为：说不知道。最坏行为：用训练知识答得很流畅。）\n")

    for guard in (False, True):
        ra, ta, ans = run_agentic(q, guard=guard)
        label = "开守卫" if guard else "不开守卫"
        print(f"  【{label}】检索 {ra} 次，工具={ta}")
        print(f"     {(ans or '')[:150]}\n")


def main():
    compare()
    demo_guard()
    print("\n" + "=" * 74)
    print("看三件事：")
    print("  ① 检索总次数差多少？（Agentic 的价值）")
    print("  ② '你好''1+1''现在几点' 这些题，传统 RAG 答得怎么样？")
    print("  ③ ★最重要★ 'C 语言释放内存' 这题，谁答对了？为什么？")
    print("=" * 74)


if __name__ == "__main__":
    main()
