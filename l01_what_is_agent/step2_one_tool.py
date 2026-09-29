"""
L01 · step2 — 给 LLM 配一只"手"：Function Calling 单次调用
==========================================================
目的：把 Function Calling 的分工看清楚——

    模型只负责"说"要调用什么（返回 tool_calls），
    真正执行的是你自己写的 Python 函数。

本步故意【不写循环】：只调一次 LLM、执行一次工具、再调一次 LLM 收尾。
所以它只能处理"一步就够"的任务。这是 step3 要修的问题。

跑法：PYTHONIOENCODING=utf-8 python l01_what_is_agent/step2_one_tool.py
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

MODEL = "glm-4-flash"


# ── 1. 工具本体：就是个普通 Python 函数 ───────────────────
def get_current_time() -> str:
    """获取当前时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── 2. 工具说明书：告诉 LLM 有这么个东西、什么时候用 ────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。当用户问'现在几点''今天几号'这类需要实时时间的问题时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]


def main():
    question = "现在几点了？"
    messages = [{"role": "user", "content": question}]
    print(f"👤 用户：{question}")

    # ── 3. 第一次调用 LLM：给它问题 + 工具说明书 ────────────
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    print("\n--- 模型第一次返回的原始结构（★重点看这里）---")
    print("content   :", repr(msg.content))
    print("tool_calls:", msg.tool_calls)

    # 模型可能不用工具、直接回答（那就不需要后面几步了）
    if not msg.tool_calls:
        print("\n模型没用工具，直接答了：", msg.content)
        return

    # ── 4. 执行工具（★ 执行的是我们，不是模型）─────────────
    call = msg.tool_calls[0]
    name = call.function.name
    # ★ arguments 是【JSON 字符串】不是 dict —— 所以要 json.loads
    args = json.loads(call.function.arguments)
    print(f"\n🤔 模型要求调用：{name}({args})")
    print(f"   arguments 的原始值：{call.function.arguments!r}")
    print(f"   arguments 的类型　：{type(call.function.arguments).__name__}")

    result = get_current_time()
    print(f"🔧 我们执行后得到：{result}")

    # ── 5. 把结果喂回去，让模型组织成自然语言答案 ───────────
    # 先把模型那一轮回复（含调用请求）记进历史，否则它不知道自己在回应什么
    messages.append(msg.model_dump())
    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    resp2 = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    print(f"\n💬 模型最终回答：{resp2.choices[0].message.content}")


if __name__ == "__main__":
    main()
