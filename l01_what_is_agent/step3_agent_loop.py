"""
L01 · step3 — 加上循环：这才叫 Agent
====================================
step2 只调了一次 LLM。step3 和它唯一的结构性区别，就是外面套了个循环：

    调 LLM（带上工具清单 + 全部历史消息）
    ├─ 它要调工具 → 执行 → 把结果塞进 messages → 回到上面再来一轮
    └─ 它不要工具了 → 输出最终答案，退出

这就是 L03 要展开的 ReAct 循环的雏形。

跑法：PYTHONIOENCODING=utf-8 python l01_what_is_agent/step3_agent_loop.py
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

MODEL = "glm-4-flash"


# ════════════════════════════════════════════════════════════
# 第 1 步：工具本体（就是普通的 Python 函数）
# ════════════════════════════════════════════════════════════
def get_current_time() -> str:
    """获取当前时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculator(expression: str) -> str:
    """计算数学表达式。

    ⚠️ 教学用 eval；生产环境绝对不行（用户输入能塞任意代码）。
        真实项目用 ast.literal_eval 或专门的计算库。
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含非法字符"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"


# ════════════════════════════════════════════════════════════
# 第 2 步：工具说明书（给 LLM 看的）
# ════════════════════════════════════════════════════════════
# description 决定 LLM 会不会选、什么时候选这个工具。
# 写得越具体（什么时候用、参数长什么样），选得越准。L04 专门讲这个。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。当用户问'现在几点''今天几号'这类需要实时时间的问题时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式。当需要精确的加减乘除时使用，不要自己心算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，如 '3 * (4 + 5)'。只能包含数字和 + - * / . ( )",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# ════════════════════════════════════════════════════════════
# 第 3 步：工具调度器 —— 把"名字 + 参数"变成真实调用
# ════════════════════════════════════════════════════════════
def execute_function(name: str, arguments: dict) -> str:
    """LLM 只负责说"要调什么"，真正的执行在这里。

    最后那个 else 分支本身就是一课：模型可能说出一个不存在的工具名。
    """
    if name == "get_current_time":
        return get_current_time()
    if name == "calculator":
        return calculator(arguments.get("expression", ""))
    return f"错误：不存在名为 {name} 的工具"


# ════════════════════════════════════════════════════════════
# 第 4 步：Agent 循环（本课核心）
# ════════════════════════════════════════════════════════════
def run_agent(question: str, max_steps: int = 6) -> str | None:
    messages = [
        {
            "role": "system",
            "content": (
                "你是严谨的助手。涉及日期和时间的问题，必须先用 get_current_time "
                "拿到准确时间再推理；涉及计算必须用 calculator，不要心算。"
            ),
        },
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        print(f"\n{'─' * 56}")
        print(f"🔄 第 {step} 步")

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # auto = 交给模型自己决定要不要用工具
        )
        msg = resp.choices[0].message

        # ── 情况 A：模型决定调用工具 ───────────────────────
        if msg.tool_calls:
            # ★ 先把模型这一轮回复（含调用请求）记进历史，
            #   否则下一轮它"不知道自己刚才要调过工具"
            messages.append(msg.model_dump())

            for call in msg.tool_calls:
                name = call.function.name
                # ★ arguments 是 JSON 字符串不是 dict，必须 loads
                args = json.loads(call.function.arguments)
                print(f"🤔 决定调用：{name}({args})")

                result = execute_function(name, args)
                print(f"🔧 执行结果：{result}")

                # ★ 把结果喂回去。role="tool" + tool_call_id 两者配对，
                #   模型才知道这个结果是回应哪次调用的
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            # 不 return，继续下一轮 —— 模型会看到工具结果再决定下一步

        # ── 情况 B：模型不要工具了 → 它给了最终答案，退出 ────
        else:
            print(f"💬 最终回答：{msg.content}")
            return msg.content

    print(f"⚠️ 走满 {max_steps} 步还没有最终答案，强制停止。")
    return None


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════
QUESTIONS = [
    # 一步工具就够
    "现在几点了？",
    # 两步：先查时间 → 再算差值。第二步的输入依赖第一步的输出
    "现在是几点？距离今天 18:00 还有多少分钟？（用计算器算）",
    # 跨两个工具
    "1234567 * 7654321 等于多少？顺便告诉我现在几点。",
]


def main():
    for q in QUESTIONS:
        print("\n" + "═" * 56)
        print(f"👤 问题：{q}")
        print("═" * 56)
        run_agent(q)


if __name__ == "__main__":
    main()
