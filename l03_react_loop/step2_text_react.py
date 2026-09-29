"""
L03 · step2 — 【经典 ReAct】不传 tools，Action 写成文本，我们自己解析
====================================================================
既然"调工具时 Thought 不出来"（见 step1），那就换个路子：

    ★ 根本不把工具交给原生 function calling。
      工具清单写在 prompt 里，让模型把"要用什么工具、什么参数"
      当成【普通文本】写出来（Thought / Action / Action Input），
      我们再从文本里解析出来、自己执行。

这就是 ReAct 原始论文的做法，也是"ReAct 不需要 function calling API"的含义。

代价很实在：你要自己写解析器，而模型不会永远守格式。

跑法：PYTHONIOENCODING=utf-8 python l03_react_loop/step2_text_react.py
"""

import json
import os
import re
from datetime import datetime

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

MODEL = "glm-4-flash"
MAX_STEPS = 8


# ── 工具（和 step1 完全一样，白名单含 %，求余数才不卡死）────────
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/%.() ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含非法字符"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"


TOOL_REGISTRY = {"get_current_time": get_current_time, "calculator": calculator}


# ── 工具清单用【纯文字】写进 prompt（不走 function calling）──
REACT_PROMPT = """你可以使用以下工具：

get_current_time()  —— 返回当前的日期和时间，不需要参数
calculator(expression: str)  —— 计算数学表达式

请严格按以下格式回复，每次只输出一步：

Thought: 你的思考（说明你为什么要这么做）
Action: 工具名
Action Input: 传给工具的参数，JSON 格式

或者，当你已经有足够信息可以回答用户时：

Thought: 你的思考
Final Answer: 给用户的最终答案

要求：
- 每次回复【只能】包含一个 Thought 加一个 Action，或一个 Thought 加一个 Final Answer。
- 不要自己编造 Observation —— Observation 会由我提供给你。"""


def parse_response(text: str):
    """从模型的自由文本里剥出结构。这是 ReAct 最脏也最关键的一段。

    返回 ("final", 答案) / ("action", 工具名, 参数字典) / ("bad", 原文)
    """
    # 1) Final Answer 优先
    fa = re.search(r"Final Answer:\s*(.*)", text, re.S)
    if fa:
        return ("final", fa.group(1).strip())

    # 2) Action —— 容忍三种写法：
    #      Action: get_current_time
    #      Action: get_current_time()
    #      Action: calculator(expression="1+2")
    m = re.search(r"Action:\s*([A-Za-z_]\w*)\s*(?:\((.*?)\))?", text, re.S)
    if not m:
        return ("bad", text)

    name = m.group(1)
    inline = (m.group(2) or "").strip()  # 写在 Action 行括号里的参数
    block = re.search(r"Action Input:\s*(.*?)(?=\n[A-Z][a-z]+:|$)", text, re.S)
    raw = block.group(1).strip() if block else inline

    args: dict = {}
    if raw:
        try:
            parsed = json.loads(raw)  # 标准情况：{"expression": "1+2"}
        except json.JSONDecodeError:
            parsed = None

        # ★ 关键：json.loads 可能【成功但返回的不是 dict】——
        #   模型写 Action Input: "13+56"，loads 会还你一个字符串。
        #   不检查类型就直接 **args 解开，会炸出
        #   "argument after ** must be a mapping, not str"。
        if isinstance(parsed, dict):
            args = parsed
        else:
            kv = re.search(r"(\w+)\s*[=:]\s*(.+)", raw)
            if kv:  # expression="1+2" 或 expression: 1+2
                args = {kv.group(1): kv.group(2).strip().strip('"').strip("'")}
            elif name == "calculator":
                args = {"expression": raw.strip().strip('"').strip("'")}
    return ("action", name, args)


def run(question: str, max_steps: int = MAX_STEPS):
    messages = [
        {"role": "system", "content": REACT_PROMPT},
        {"role": "user", "content": question},
    ]
    trace = []
    bad_format = 0

    for step in range(1, max_steps + 1):
        print(f"\n{'━' * 56}\n🔄 第 {step} 轮")

        resp = client.chat.completions.create(model=MODEL, messages=messages)
        text = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": text})

        parsed = parse_response(text)

        # ── 情况 A：给出最终答案 ──────────────────────────
        if parsed[0] == "final":
            thought = re.search(r"Thought:\s*(.*?)(?=\nFinal Answer:|$)", text, re.S)
            print(f"   💭 Thought: {thought.group(1).strip() if thought else '(没写)'}")
            print(f"   ✅ Final Answer: {parsed[1]}")
            trace.append(("Final Answer", parsed[1]))
            return trace, parsed[1]

        # ── 情况 B：格式不守 ──────────────────────────────
        if parsed[0] == "bad":
            bad_format += 1
            print(f"   ⚠️ 没解析出 Action / Final Answer。原文：{text[:80]!r}")
            trace.append(("格式错误", text[:60].replace("\n", " ")))
            messages.append(
                {
                    "role": "user",
                    "content": "格式不对。请严格按 Thought / Action / Action Input 回复。",
                }
            )
            continue

        # ── 情况 C：要调工具 ──────────────────────────────
        _, name, args = parsed
        thought = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.S)
        print(f"   💭 Thought: {thought.group(1).strip() if thought else '(没写)'}")

        if name not in TOOL_REGISTRY:
            obs = f"错误：没有名为 {name} 的工具。可用工具：{list(TOOL_REGISTRY)}"
        else:
            try:
                obs = str(TOOL_REGISTRY[name](**args))
            except Exception as e:
                obs = f"参数错误：{e}"

        print(f"   🔧 Action: {name}({args})")
        print(f"   👁️ Observation: {obs}")
        trace.append((f"Action {name}", f"{args} -> {obs}"))
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    print(f"\n⚠️ 走满 {max_steps} 轮仍未结束。")
    trace.append(("撞上限", ""))
    return trace, None


def main():
    print("=" * 60)
    print("L03 · step2 — 经典 ReAct（Action 靠文本解析）")
    print("=" * 60)
    q = "现在几点？请把当前小时数和分钟数相加，再用结果除以 7，告诉我余数是多少。"
    print(f"\n👤 {q}")

    trace, ans = run(q)

    print("\n" + "=" * 60)
    print("📊 完整推理链（这就是 ReAct 的价值：每一步都看得见）")
    print("=" * 60)
    for i, (kind, body) in enumerate(trace, 1):
        print(f"   {i:>2}. [{kind}] {body}")
    print(f"\n   ★ 最终答案：{ans}")


if __name__ == "__main__":
    main()
