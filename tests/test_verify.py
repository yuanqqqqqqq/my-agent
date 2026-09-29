"""校验层的单元测试 —— 纯函数，不联网、不要 API Key。

    python tests/test_verify.py        # 直接跑
    pytest tests/ -q                   # 或用 pytest

为什么这个文件值得存在：
  校验层是这个项目的立论依据（"用代码验证，而不是再叫一个模型来盖章"）。
  如果校验层自己判错，整个项目的结论就不成立了 —— 而它判错的代价比模型幻觉
  更隐蔽：日志上看是"校验生效了"，实际是在拒绝正确答案或者在放行假答案。

  ⚠️ 这里测的每一条都对应一次真实事故，不是想出来的边界。见各组注释。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_agent.verify import (  # noqa: E402
    _is_declined,
    _norm_source,
    check_citations,
    check_coverage,
    check_numbers,
    check_scope,
    check_substance,
    verify,
)

# 工具返回的原文：报告里的数字必须能在这里找到出处
TOOL_LOG = (
    "[来源: kb://fastapi_docs/request-files.md]\n"
    "上传文件用 UploadFile。默认最大 1 个文件，单文件上限 1048576 字节。\n\n"
    "[来源: https://fastapi.tiangolo.com]\n"
    "FastAPI is a modern web framework. 安装：uv add fastapi\n"
)
REAL = {"kb://fastapi_docs/request-files.md", "https://fastapi.tiangolo.com"}

GOOD_POINT = ("默认情况下路径参数是字符串类型，所以 /items/3 返回 item_id 为 \"3\"，"
              "原文：path parameters are strings by default")


def _report(points=None, summary="FastAPI 简介", sources=None):
    return {"summary": summary, "points": points or [], "sources": sources or []}


def _pt(point=GOOD_POINT, source="kb://fastapi_docs/request-files.md"):
    return {"point": point, "source": source}


# ══════════════════════════════════════════════════════
# 三态判定（拒答 vs 作答）—— 本项目踩过最贵的一个坑
# ══════════════════════════════════════════════════════
def test_decline_uses_structure_not_wording():
    """拒答与否看【有没有要点】，不看它用了哪句话。

    ★ 这是实测事故的回归护栏。旧实现是关键词表
      `re.search("(没有找到|未找到|资料不足|无法确认|没有相关资料)", text)`，
      拿 8 个真实样本来测【错判 6 个】。下面三句都是老实的拒答，
      但旧实现一句都认不出来 —— 因为它换个说法就废了。
    """
    for wording in [
        "没有找到相关资料",                        # 这句旧实现认得
        "无法找到关于公司下周团建安排的资料。",        # 旧实现认不出（表里只有"无法确认"）
        "这个问题不需要查任何资料，也没有可引用的来源。",  # 旧实现认不出
        "资料库中没有这个信息",                     # 旧实现认不出
        "抱歉，我无法回答这个问题",                  # 旧实现认不出
    ]:
        assert _is_declined(_report(points=[], summary=wording)) is True, (
            f"这句是合法拒答，不该被判成作答：{wording}"
        )


def test_answer_containing_decline_phrase_is_not_a_decline():
    """真答案的子句里带了「没有找到」字样，仍然是答案，必须走完整校验。

    ★ 同一事故的【另一半】。旧实现只看子串有没有出现，于是这句货真价实的
      答案被当成"拒答"直接放行、一个校验都不做 —— 漏放比误杀更危险，
      因为它在日志上看起来完全正常。
    """
    r = _report(points=[_pt("FastAPI 用 File 和 Form 处理上传，"
                            "不过官方文档里没有找到更细的说明")])
    assert _is_declined(r) is False, "有要点就是作答，子句里出现什么词都不影响"


def test_decline_needs_having_tried_web_search():
    """拒答分两种：真找过了 / 只查了一个来源就放弃。后者要打回。"""
    declined = _report(points=[], summary="没有找到相关资料")

    ok, problems = verify(declined, retrieved=set(), tool_output="",
                          required_points=[], tools_called={"search_kb"})
    assert not ok, "只查了本地库就下结论'没有资料'，应该被打回"
    assert any("search_web" in p for p in problems), problems

    ok, problems = verify(declined, retrieved=set(), tool_output="",
                          required_points=[], tools_called={"search_kb", "search_web"})
    assert ok, f"真的找过了，拒答应当放行：{problems}"


def test_decline_passes_even_without_points_or_sources():
    """拒答没有要点、没有引用 —— 全是正常的，不该被任何校验拦下。"""
    ok, problems = verify(_report(points=[], summary="没有找到相关资料"),
                          retrieved=set(), tool_output="", required_points=["上传文件"])
    assert ok, f"合格的拒答不该被拦：{problems}"


# ══════════════════════════════════════════════════════
# 来源归一化
# ══════════════════════════════════════════════════════
def test_norm_source_strips_wrapper_and_slash():
    """模型常把整个 [来源: xxx] 连方括号抄进来 —— 那是格式不符，不是编造来源。"""
    assert _norm_source("[来源: kb://a/b.md]") == "kb://a/b.md"
    assert _norm_source("来源：kb://a/b.md") == "kb://a/b.md"
    assert _norm_source("  https://x.com/  ") == "https://x.com"


# ══════════════════════════════════════════════════════
# ① 引用真实性
# ══════════════════════════════════════════════════════
def test_ghost_citation_is_caught():
    problems = check_citations(_report(points=[_pt(source="https://fake.com")]), REAL)
    assert problems and "fake.com" in problems[0], problems


def test_citation_wrapper_is_not_treated_as_fake():
    """格式不符 ≠ 编造来源。不归一化就会连环误杀（实测被拦了 3 次）。"""
    problems = check_citations(
        _report(points=[_pt(source="[来源: kb://fastapi_docs/request-files.md]")]), REAL
    )
    assert not problems, f"归一化后应当是合法来源，不该被拦：{problems}"


def test_report_without_any_source_is_caught():
    problems = check_citations(_report(points=[_pt(source="")]), REAL)
    assert problems and "来源" in problems[0], problems


# ══════════════════════════════════════════════════════
# ② 数字可溯源
# ══════════════════════════════════════════════════════
def test_number_without_source_is_caught():
    """治的是"模型编个数字去算"这个病。"""
    r = _report(points=[_pt("性能据称提升了 4837%")])
    problems = check_numbers(r, TOOL_LOG)
    assert problems and "4837" in problems[0], problems


def test_number_present_in_tool_output_passes():
    r = _report(points=[_pt("单文件上限是 1048576 字节")])
    assert not check_numbers(r, TOOL_LOG), "这个数字工具返回里有，不该被拦"


# ══════════════════════════════════════════════════════
# ③ 覆盖度
# ══════════════════════════════════════════════════════
def test_uncovered_plan_point_is_caught():
    problems = check_coverage(_report(points=[_pt()]), ["文件上传怎么实现"])
    assert problems, "计划里的要点没覆盖到，应该被拦"


def test_covered_plan_point_passes():
    r = _report(points=[_pt("文件上传用 File 和 Form 参数")])
    assert not check_coverage(r, ["文件上传"]), "提到了就不该拦"


# ══════════════════════════════════════════════════════
# ④ 空壳检测
# ══════════════════════════════════════════════════════
def test_report_with_no_points_is_caught_by_substance():
    problems = check_substance(_report(points=[]))
    assert problems and "要点" in problems[0], problems


def test_all_thin_points_are_caught():
    problems = check_substance(_report(points=[_pt("它是框架"), _pt("很快")]))
    assert problems, "所有要点都太短，看不出依据，应该被拦"


def test_one_solid_point_is_enough():
    """只要有要点是扎实的就不拦 —— 不追求每一条都长，避免误杀。"""
    assert not check_substance(_report(points=[_pt("它是框架"), _pt()]))


# ══════════════════════════════════════════════════════
# ⑥ 限定词：私有问题不能用公开资料搪塞
# ══════════════════════════════════════════════════════
def test_private_question_answered_from_public_sources_is_caught():
    """实测回归：问"我们公司下周团建安排"，报告引用真 100%，但引的全是
    "团建活动怎么策划"这类公开文章 —— 来源是真的，但和"我们公司"毫无关系。"""
    r = _report(points=[_pt(source="https://zhihu.com/how-to-plan-teambuilding")],
                sources=["https://zhihu.com/how-to-plan-teambuilding"])
    problems = check_scope(r, "我们公司下周的团建安排是什么？", set())
    assert problems and "私有" in problems[0], problems


def test_private_question_with_decline_passes():
    r = _report(points=[], summary="没有找到针对贵方的具体信息")
    assert not check_scope(r, "我们公司下周的团建安排是什么？", set())


def test_public_question_is_not_scope_checked():
    r = _report(points=[_pt()])
    assert not check_scope(r, "FastAPI 怎么处理文件上传？", REAL), "公开问题不该触发这条"


# ══════════════════════════════════════════════════════
# 端到端
# ══════════════════════════════════════════════════════
def test_valid_report_passes_all_checks():
    r = _report(points=[_pt("文件上传用 UploadFile，单文件上限是 1048576 字节，"
                            "原文：单文件上限 1048576 字节")],
                sources=["kb://fastapi_docs/request-files.md"])
    ok, problems = verify(r, retrieved=REAL, tool_output=TOOL_LOG,
                          required_points=["文件上传"], tools_called={"search_kb"},
                          topic="FastAPI 怎么处理文件上传？")
    assert ok, f"完整报告应当通过：{problems}"


def test_fake_source_in_answer_fails():
    r = _report(points=[_pt(source="https://fake.com")], sources=["https://fake.com"])
    ok, problems = verify(r, retrieved=REAL, tool_output=TOOL_LOG,
                          required_points=[], tools_called={"search_kb"},
                          topic="FastAPI 怎么处理文件上传？")
    assert not ok and problems, "编造来源必须拦下"


# ══════════════════════════════════════════════════════


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
