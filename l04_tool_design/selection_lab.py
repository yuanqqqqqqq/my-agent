"""
L04 · 工具选择实验台
====================
一个"测量仪器"：给定【工具定义】和【问题】，看模型第一次选了哪个工具。

用法：把工具定义的某一个部分（描述 / 名字 / 参数名 / 顺序）改掉，重跑，
      看选择结果变不变。一次只改一个变量，才能知道是哪个信号在起作用。

跑法：PYTHONIOENCODING=utf-8 python l04_tool_design/selection_lab.py
"""

import os

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

MODEL = "glm-4-flash"
N = 8          # 每个对照跑几次
Q_REVERSE = "把 'hello' 这个字符串反转过来（倒序输出）"


# ════════════════════════════════════════════════════════════
# 造工具的小助手 + 只跑第一轮的探测器
# ════════════════════════════════════════════════════════════
def tool(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


def pick_once(question, spec):
    """只问一轮，返回模型第一次选的工具名（不执行、不循环）。

    ★ 为什么只跑第一轮：我们要测的是"选工具"这个动作本身，
      后面几轮受前一次结果影响，会污染结论。
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        tools=spec,
        tool_choice="auto",
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        return msg.tool_calls[0].function.name
    return "(没调工具)"


def experiment(title, question=Q_REVERSE, expect=None):
    """跑一组对照：打印每个工具被选中几次。

    ⚠️ 把 expect 设成正确答案会误导你 —— 我们的目的不是"跑对"，
       是【看信号：改了什么，选择就变了】。
    """
    def run(spec):
        tally = {}
        for _ in range(N):
            name = pick_once(question, spec)
            tally[name] = tally.get(name, 0) + 1
        return tally

    return run


def show(title, tally, expect=None, note=""):
    print(f"\n【{title}】")
    if note:
        print(f"   {note}")
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        mark = ""
        if expect:
            mark = "✅ " if k == expect else "❌ "
        print(f"   {mark}{k}: {v}/{N}")


# ════════════════════════════════════════════════════════════
# 工具定义：同一对工具（反转 / 数长度），只改【一个变量】
# ════════════════════════════════════════════════════════════
DESC_REV = "把字符串反转（首尾倒过来）。当用户说'反转''倒序''逆序输出'时使用。"
DESC_LEN = "计算字符串的字符数（长度）。当用户问'XX有几个字''XX多长'时使用。"
DESC_NONE = "操作文本。"

SP = lambda p: {"p": {"type": "string"}}          # noqa: E731  参数都叫 p、都是 string


def A_all_good():
    """基线：名字、描述都有信息。"""
    return [tool("string_reverse", DESC_REV, SP("p"), ["p"]),
            tool("string_length", DESC_LEN, SP("p"), ["p"])]


def B_desc_only():
    """只有描述有信息（名字中性）。"""
    return [tool("tool_alpha", DESC_REV, SP("p"), ["p"]),
            tool("tool_beta", DESC_LEN, SP("p"), ["p"])]


def C_conflict():
    """名字和描述【打架】：名字说 reverse 的描述在讲数长度，反之亦然。"""
    return [tool("string_reverse", DESC_LEN, SP("p"), ["p"]),
            tool("string_length", DESC_REV, SP("p"), ["p"])]


def D_name_only():
    """只有名字有信息（描述完全相同且无意义）—— 看名字能不能单独扛。"""
    return [tool("string_reverse", DESC_NONE, SP("p"), ["p"]),
            tool("string_length", DESC_NONE, SP("p"), ["p"])]


def E_paramname_only():
    """名字中性 + 描述无意义，只有【参数名】有语义。"""
    return [tool("tool_alpha", DESC_NONE, {"text_to_reverse": {"type": "string"}}, ["text_to_reverse"]),
            tool("tool_beta", DESC_NONE, {"text_to_count": {"type": "string"}}, ["text_to_count"])]


def F_nothing():
    """名字、描述、参数名 —— 全都没有信息。"""
    return [tool("tool_alpha", DESC_NONE, SP("p"), ["p"]),
            tool("tool_beta", DESC_NONE, SP("p"), ["p"])]


def main():
    print("=" * 64)
    print("L04 · 工具选择实验台")
    print(f"问题固定为：{Q_REVERSE}")
    print(f"每组跑 {N} 次。★ 一次只改一个变量。")
    print("=" * 64)

    run = experiment("")

    cases = [
        ("A 基线：名字 + 描述都有信息", A_all_good, "string_reverse"),
        ("B 只有【描述】有信息（名字中性）", B_desc_only, "tool_alpha"),
        ("C 名字与描述【打架】（问反转）", C_conflict, "tool_length? 看谁赢"),
        ("D 只有【名字】有信息（描述全同且无意义）", D_name_only, "string_reverse"),
        ("E 只有【参数名】有信息", E_paramname_only, "tool_alpha"),
        ("F 三个信号全都没有信息", F_nothing, None),
    ]
    for title, fn, expect in cases:
        show(title, run(fn()), expect)

    print("\n" + "=" * 64)
    print("怎么读这些结果：")
    print("  · 哪一组还能选对，说明【那一组唯一有信息的那个信号】是有效的。")
    print("  · C 组谁赢，说明【描述】和【名字】谁优先级更高。")
    print("  · F 组是最有信息量的一组：什么信号都没有时，模型会怎么办？")
    print("=" * 64)


if __name__ == "__main__":
    main()
