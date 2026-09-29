"""
能力评测 —— 有校验层 vs 无校验层
==================================
这是本项目的立论依据：**用数字证明校验层有用**，而不是说"我设计了一个校验层"。

评测集分三类，覆盖这个项目能遇到的全部情况：
  A 本地知识库有答案    → 应该查本地库，给出带来源的正确报告
  B 只有联网有答案      → 本地库查不到，应该转向联网，而不是直接放弃
  C 两个来源都没有      → 应该【如实拒答】，绝不编造

指标（全部可判定，不依赖人去打分）：
  · 产出报告      是否给出了报告
  · 校验结论      有校验层时：通过 / 被打回
  · 打回次数
  · 工具调用      用了几个工具（成本代理指标）
  · 引用真实率    报告里的来源，有多少比例是【本次真实检索到】的
  · 行为正确      期望作答的作对了 / 期望拒答的拒答了
  · 耗时

跑法：
  PYTHONIOENCODING=utf-8 python research_agent/eval.py              # 只跑"有校验层"
  PYTHONIOENCODING=utf-8 python research_agent/eval.py --no-verify  # 只跑"无校验层"
"""

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.agent import research                      # noqa: E402
from research_agent.verify import _is_declined, _norm_source   # noqa: E402

# ════════════════════════════════════════════════════════════
# 评测集
# ════════════════════════════════════════════════════════════
# (主题, 类别, 期望行为)   expect: "answer" = 该作答；"decline" = 该拒答
TOPICS = [
    ("FastAPI 怎么处理文件上传？",              "A 本地库有",   "answer"),
    ("FastAPI 的依赖注入是怎么工作的？",          "A 本地库有",   "answer"),
    ("FastAPI 里怎么用 Pydantic 做请求体校验？",   "A 本地库有",   "answer"),
    ("RAG 检索增强生成的核心思路是什么？",         "B 只有联网有", "answer"),
    ("大模型的幻觉问题有哪些主流缓解方法？",        "B 只有联网有", "answer"),
    ("我们公司下周的团建安排是什么？",             "C 都不该有",   "decline"),
    ("我昨天中午吃了什么？",                     "C 都不该有",   "decline"),
]


# ════════════════════════════════════════════════════════════
# 审计：不看模型说了什么，只看报告和真实检索结果对不对得上
# ════════════════════════════════════════════════════════════
def audit(report: dict | None, retrieved_set: list[str]):
    if not report:
        return {"declined": None, "cited": 0, "real_rate": 0.0, "fake": []}
    declined = _is_declined(report)

    cited = set(report.get("sources") or [])
    for p in report.get("points") or []:
        if s := p.get("source"):
            cited.add(s)
    cited = {_norm_source(s) for s in cited if s}
    real = {_norm_source(s) for s in retrieved_set}
    fake = sorted(c for c in cited if c not in real)
    rate = (len(cited) - len(fake)) / len(cited) if cited else (1.0 if declined else 0.0)
    return {"declined": declined, "cited": len(cited), "real_rate": rate, "fake": fake}


def main():
    use_verify = "--no-verify" not in sys.argv
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    label = "有校验层" if use_verify else "无校验层"
    print("=" * 88)
    print(f"能力评测 · {label}   （模型 {__import__('research_agent.agent', fromlist=['MODEL']).MODEL}）")
    print("=" * 88)

    rows = []
    for topic, cat, expect in (TOPICS[:limit] if limit else TOPICS):
        t0 = time.time()
        try:
            report, stats = research(topic, use_verify=use_verify, quiet=True)
            err = ""
        except Exception as e:
            report, stats, err = None, {}, f"{type(e).__name__}: {e}"

        a = audit(report, stats.get("retrieved_set", []))
        # ★ 标签必须分清三种"失败"，它们是完全不同的问题：
        #    ① 压根没产出报告（撞步数上限）—— 是"没做完"，不是"编造"
        #    ② 该作答却拒答 —— 是"过早放弃"
        #    ③ 作答了但引用有假 —— 才是"编造"
        #   ⚠️ 我第一版把 ① 和 ③ 归成一类，报了"❌ 无据/编造"，是错的。
        #      这又是"检测口径"问题 —— 标签说的事和真实发生的事不是一回事。
        if err:
            behavior = "❌ 报错"
        elif not report:
            behavior = "❌ 未产出报告(撞上限)"
        elif expect == "decline":
            behavior = "✅ 老实拒答" if a["declined"] else "❌ 该拒答却没拒"
        elif a["fake"]:
            behavior = f"❌ 编造来源 {a['fake'][:2]}"
        elif a["declined"]:
            behavior = "⚠️ 该作答却拒答"
        else:
            behavior = "✅ 有据作答"

        rows.append({
            "topic": topic, "cat": cat, "expect": expect,
            "has_report": report is not None,
            "verified": stats.get("verified"),
            "rewrites": stats.get("rewrites", 0),
            "tools": stats.get("tools", 0),
            "steps": stats.get("steps", 0),
            "real_rate": a["real_rate"], "fake": a["fake"],
            "behavior": behavior,
            "elapsed": round(time.time() - t0, 1),
            "err": err,
        })
        print(f"  [{len(rows)}/{len(TOPICS[:limit] if limit else TOPICS)}] {topic[:34]:<36} "
              f"{behavior}  耗时 {rows[-1]['elapsed']}s")

    # ── 明细表 ──
    print("\n" + "=" * 88)
    print(f"{'主题':<34}{'类别':<14}{'校验':<8}{'打回':<6}{'工具':<6}{'引用真实率':<12}{'行为'}")
    print("-" * 88)
    for r in rows:
        v = {True: "✅通过", False: "❌不过", None: "—"}[r["verified"]]
        print(f"{r['topic'][:32]:<34}{r['cat']:<14}{v:<8}{r['rewrites']:<6}{r['tools']:<6}"
              f"{r['real_rate']*100:>6.0f}%     {r['behavior']}")

    # ── 汇总 ──
    n = len(rows)
    print("-" * 88)
    ok = sum(1 for r in rows if r["behavior"].startswith("✅"))
    declined_ok = sum(1 for r in rows if r["expect"] == "decline" and r["behavior"].startswith("✅"))
    declined_n = sum(1 for r in rows if r["expect"] == "decline")
    avg_tools = sum(r["tools"] for r in rows) / n
    avg_rate = sum(r["real_rate"] for r in rows) / n
    print(f"  行为正确率    {ok}/{n}")
    print(f"  拒答正确率    {declined_ok}/{declined_n}   （C 类题：两个来源都没有，最该老实）")
    print(f"  平均工具调用  {avg_tools:.1f} 次/题")
    print(f"  平均引用真实率 {avg_rate*100:.0f}%")
    if use_verify:
        passed = sum(1 for r in rows if r["verified"] is True)
        print(f"  校验通过率    {passed}/{n}")
        print(f"  总打回次数    {sum(r['rewrites'] for r in rows)}")
    print(f"  总耗时        {sum(r['elapsed'] for r in rows):.0f}s")

    # 落盘，供两次模式对比
    out = f"research_agent/.eval_{'verify' if use_verify else 'noverify'}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n  明细已写入 {out}")


if __name__ == "__main__":
    main()
