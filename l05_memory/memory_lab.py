"""
L05 · 记忆：三种窗口策略 + 两个量化实验
========================================
核心认知：模型自己没有记忆。你传多少历史，它就"记得"多少。

三个东西：
  ① 三种窗口管理策略：全保留 / 截断 / 摘要压缩
  ② 实验 A：截断窗口到底要多大才够？（扫描 keep_last）
  ③ 实验 B：摘要能保住多少信息？—— 关键变量是【prompt 有没有点名要保什么】

跑法：PYTHONIOENCODING=utf-8 python l05_memory/memory_lab.py
"""

import os

from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()
client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

MODEL = "glm-4-flash"


def chat(messages: list) -> str:
    """基础对话：传入完整 messages（含历史），返回回答。

    ★ 记忆的全部秘密就在这一行的入参上 —— messages 里有什么，它就"记得"什么。
    """
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content


# ════════════════════════════════════════════════════════════
# 三种窗口管理策略
# ════════════════════════════════════════════════════════════
def keep_all(messages: list) -> list:
    """策略 1：全保留。信息最全，但对话越长越贵越慢。"""
    return messages.copy()


def truncate_messages(messages: list, keep_last: int = 6) -> list:
    """策略 2：截断（滑动窗口）。只保留 system + 最近 keep_last 条。

    ⚠️ keep_last 数的是【消息条数】，一轮对话 = user + assistant = 2 条。
       所以 keep_last=6 约等于最近 3 轮。
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    others = [m for m in messages if m.get("role") != "system"]
    return system_msgs + (others[-keep_last:] if keep_last > 0 else [])


SUMMARY_PROMPT_VAGUE = "请把下面这段对话历史压缩成一段简洁的摘要，保留关键信息，不超过 100 字：\n\n"
SUMMARY_PROMPT_EXPLICIT = (
    "请把下面这段对话历史压缩成一段简洁的摘要。"
    "必须逐条保留以下具体信息，不要只写类别：用户的名字、具体喜好、具体过敏源、"
    "具体日期、具体住址、尚未决定的问题。不超过 100 字：\n\n"
)


def summarize_history(old_messages: list, prompt_template: str) -> str:
    """策略 3：用模型把旧历史压成摘要。

    ★ 注意 prompt_template 是个参数 —— 实验 B 就是靠换它来对比的。
    """
    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in old_messages if m.get("content")
    )
    out = chat([{"role": "user", "content": prompt_template + history_text}])
    return "[之前的对话摘要] " + out


def compress_with_summary(messages: list, keep_recent: int = 4,
                          prompt_template: str = SUMMARY_PROMPT_EXPLICIT) -> list:
    """摘要压缩 = 旧历史压成一条摘要 + 最近几轮原样保留。"""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    others = [m for m in messages if m.get("role") != "system"]
    if len(others) <= keep_recent:
        return messages.copy()
    summary = summarize_history(others[:-keep_recent], prompt_template)
    return system_msgs + [{"role": "user", "content": summary}] + others[-keep_recent:]


def estimate_tokens(messages: list) -> int:
    """粗略估算 token（中文约 2 字符/token）。教学够用，真实场景用 tokenizer。"""
    return sum(len(m.get("content", "") or "") for m in messages) // 2


# ════════════════════════════════════════════════════════════
# 实验 1：多轮对话基线（证明"记忆 = messages"）
# ════════════════════════════════════════════════════════════
def exp1_multi_turn():
    print("\n" + "═" * 64)
    print("实验 1：多轮对话 —— Agent 真的「记得」吗？")
    print("═" * 64)

    messages = [{"role": "system", "content": "你是友好的助手，请简短回答。"}]
    for i, user_input in enumerate(
        ["我叫张三，我喜欢蓝色。", "今天我心情不太好。", "我还记得我的名字和喜欢的颜色吗？"], 1
    ):
        print(f"\n第 {i} 轮 🙋 {user_input}")
        messages.append({"role": "user", "content": user_input})
        reply = chat(messages)
        print(f"        🤖 {reply[:80]}")
        messages.append({"role": "assistant", "content": reply})

    print("\n👉 它能答出'张三''蓝色'，唯一原因是历史都在 messages 里。")
    print("   把那两行 messages.append 删掉，它立刻就全忘了。")


# ════════════════════════════════════════════════════════════
# 实验 2：截断窗口扫描 —— 到底要多大才够？
# ════════════════════════════════════════════════════════════
def exp2_truncate_scan(chatty_rounds: int = 8):
    print("\n" + "═" * 64)
    print("实验 2：截断窗口扫描 —— keep_last 要多大才记得住名字？")
    print("═" * 64)

    history = [{"role": "system", "content": "你是助手，请简短回答。"},
               {"role": "user", "content": "我叫李四，请记住我的名字。"},
               {"role": "assistant", "content": "好的李四，我记住了。"}]
    for i in range(1, chatty_rounds + 1):
        history.append({"role": "user", "content": f"随便聊一句，第 {i} 句。"})
        history.append({"role": "assistant", "content": f"好的，第 {i} 句收到。"})

    others = [m for m in history if m["role"] != "system"]
    print(f"历史共 {len(history)} 条（{len(others)} 条非 system）。名字埋在第 1 条。\n")

    for keep in [2, 4, 6, 8, 10, 12, 14, 16, len(others)]:
        kept = truncate_messages(history, keep_last=keep)
        msgs = kept + [{"role": "user", "content": "我叫什么名字？"}]
        ans = chat(msgs)
        ok = "李四" in ans
        print(f"  keep_last={keep:>3}（留 {len(kept)+1:>2} 条）  {'✅ 记住' if ok else '❌ 失忆'}  {ans[:30]}")

    print("\n👉 结论不是'keep_last 要设多大'，而是：")
    print("   截断窗口够不够，取决于【关键信息离现在有多远】，没有通用安全值。")


# ════════════════════════════════════════════════════════════
# 实验 3：摘要 prompt 的 A/B —— 决定保留率的是什么？
# ════════════════════════════════════════════════════════════
FACTS = {                       # 关键检测词（不用整句，防措辞变化导致误判）
    "名字": ["李四"],
    "颜色": ["蓝色", "喜蓝", "偏好蓝色", "喜欢蓝色"],
    "过敏": ["花生"],
    "生日": ["3月15日", "3 月 15 日", "三月十五", "3-15", "3/15"],
    "住址": ["杭州"],
    "未决问题": ["MacBook", "macbook"],
}

CONVO = """用户: 我叫李四。
助手: 你好李四。
用户: 我最喜欢的颜色是蓝色。
助手: 蓝色很好。
用户: 提醒你一下，我对花生过敏。
助手: 记下了，花生。
用户: 我的生日是 3 月 15 日。
助手: 好的，3 月 15 日。
用户: 我住在杭州。
助手: 杭州不错。
用户: 我最近在纠结要不要买 MacBook。
助手: 可以多对比一下。
用户: 暂时不决定。
助手: 好的。"""


def check_facts(text: str) -> tuple[list, list]:
    """按【关键词】判断事实有没有被保留。

    ⚠️ 别用整句匹配（"要不要买 MacBook"）—— 摘要一旦改写成"是否买 MacBook"，
       字面匹配就会误报"丢了"。我第一版就栽在这上面。
       ★ 指标口径决定诊断，这是你 RAG Step 8 学过的。
    """
    survived, lost = [], []
    for name, keys in FACTS.items():
        (survived if any(k in text for k in keys) else lost).append(name)
    return survived, lost


def exp3_summary_ab(runs: int = 3):
    print("\n" + "═" * 64)
    print("实验 3：摘要 prompt 的 A/B —— 谁来决定「保住了什么」？")
    print("═" * 64)
    print(f"同一段对话（埋 {len(FACTS)} 条事实），只换 prompt，各跑 {runs} 次。\n")

    for label, tmpl in [("A 泛泛说「保留关键信息」", SUMMARY_PROMPT_VAGUE),
                        ("B 点名「逐条保留具体值」", SUMMARY_PROMPT_EXPLICIT)]:
        print(f"  【{label}】")
        for i in range(1, runs + 1):
            s = summarize_history(
                [{"role": "user", "content": CONVO}], tmpl
            ).replace("[之前的对话摘要] ", "")
            sv, ls = check_facts(s)
            print(f"    [{i}] 保住 {len(sv)}/{len(FACTS)}  丢了 {ls}")
            print(f"         {s[:95]}")
        print()

    print("👉 看出来的关键在于：变量不是「压缩」这个动作，而是 prompt 有没有【点名】。")
    print("   泛泛说'保留关键信息'，模型会把'花生过敏/蓝色/3月15日/杭州'抽象成'个人喜好'。")
    print("   ★ 压缩的损失不是'丢信息'，是【把具体值换成类别名】。")


def exp4_repeat_compress(rounds: int = 4):
    print("\n" + "═" * 64)
    print("实验 4：反复压缩会怎样？（长对话里会真实发生）")
    print("═" * 64)
    text = CONVO
    for i in range(1, rounds + 1):
        text = chat([{"role": "user", "content": SUMMARY_PROMPT_VAGUE + text}])
        sv, ls = check_facts(text)
        print(f"  第 {i} 次压缩后（{len(text)} 字）保住 {len(sv)}/{len(FACTS)}：{sv}")
        print(f"      {text[:95]}")
    print("\n👉 具体值一旦被抽象掉，后面再怎么压都救不回来。")


# ════════════════════════════════════════════════════════════
# 交互模式：自己和它聊
# ════════════════════════════════════════════════════════════
def interactive_mode():
    print("\n" + "═" * 64)
    print("交互模式：输入 exit 退出。先告诉它你的名字和喜好，聊几轮再问它。")
    print("═" * 64)
    messages = [{"role": "system", "content": "你是友好的助手，请简短回答。"}]
    while True:
        user = input("\n🙋 ").strip()
        if user.lower() in ("exit", "quit", ""):
            break
        messages.append({"role": "user", "content": user})
        reply = chat(messages)
        print(f"🤖 {reply}")
        messages.append({"role": "assistant", "content": reply})
    print(f"\n本轮共 {len(messages)} 条消息，约 {estimate_tokens(messages)} tokens。")


def main():
    print("=" * 64)
    print("L05 · 记忆：三种窗口策略 + 两个量化实验")
    print("=" * 64)

    exp1_multi_turn()
    exp2_truncate_scan()
    exp3_summary_ab()
    exp4_repeat_compress()

    print("\n" + "=" * 64)
    print("三种策略的取舍（记住这张表就够了）：")
    print("  全保留：信息最全，但越长越贵越慢")
    print("  截断　：简单省 token，但关键信息一被切掉就永久失忆")
    print("  摘要　：能保关键信息，但保多少【取决于你的 prompt 有没有点名】")
    print("=" * 64)
    print("\n想自己聊：把下面这行取消注释，重跑。")
    print("   # interactive_mode()")


if __name__ == "__main__":
    main()
