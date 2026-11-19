"""结果呈现：原始观测、可行推断与结论范围分列，不把区间压成点值。"""
from __future__ import annotations


import pandas as pd

from .network import EDGE_ORDER, METERED_EDGES, UNMETERED_EDGES


def fmt_interval(lo: float, hi: float, digits: int = 2) -> str:
    if lo == hi:
        return f"{lo:.{digits}f}"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def balance_diagnosis_table(balance_df: pd.DataFrame) -> pd.DataFrame:
    """逐日平衡诊断：冲突日、节点松弛、未计量流量可行区间宽度。"""
    cols = ["day", "status", "retrofit", "maintenance", "load"]
    out = balance_df[cols].copy()
    out["day"] = pd.to_datetime(out["day"]).dt.date
    for node in ("HP", "MP", "LP"):
        c = f"slack_{node}"
        if c in balance_df:
            out[c] = balance_df[c].round(2)
    for e in UNMETERED_EDGES:
        out[f"{e}_区间"] = [
            fmt_interval(lo, hi) for lo, hi in
            zip(balance_df[f"lo_{e}"], balance_df[f"hi_{e}"])
        ]
    return out


def flow_comparison_table(evals: dict) -> pd.DataFrame:
    """基线与各方案的边流量点值/区间对照（每日均值，吨）。"""
    rows = []
    for e in EDGE_ORDER:
        row = {"edge": e, "metered": e in METERED_EDGES}
        for pid, ev in evals.items():
            f = ev["flows"][e]
            row[pid] = fmt_interval(f["lo"], f["hi"])
        rows.append(row)
    return pd.DataFrame(rows)


def money_table(result: pd.DataFrame) -> pd.DataFrame:
    """费用与排放变化表（正=多支出/多排放）。"""
    cols = [
        "proposal_id", "coal_status", "conflict_days", "induced_conflict_days",
        "mean_slack_t", "device_add_kwh",
        "delta_import_kwh_lo", "delta_import_kwh_hi",
        "delta_boiler_t_lo", "delta_boiler_t_hi",
        "delta_coal_t_lo", "delta_coal_t_hi",
        "delta_cost_yuan_lo", "delta_cost_yuan_hi",
        "delta_emission_kg_lo", "delta_emission_kg_hi",
    ]
    cols = [c for c in cols if c in result.columns]
    out = result[cols].copy().rename(columns={
        "conflict_days": "不可平衡日数",
        "induced_conflict_days": "方案诱发不可平衡日",
        "mean_slack_t": "日均HP松弛t",
        "proposal_id": "方案",
        "coal_status": "煤项状态",
        "device_add_kwh": "设备增电kWh",
    })
    return out.round(2)


def scenario_sweep(pack, scenarios: list[str] | None = None) -> pd.DataFrame:
    """同一物理反事实在不同外部价格/排放假设下的结果，区分测量与价格来源。"""
    from .eval import run_review

    scenarios = scenarios or list(pack.external.scenario)
    frames = []
    for sc in scenarios:
        result, _ = run_review(pack, scenario=sc)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def text_report(pack, scenario: str, result: pd.DataFrame, diag: dict,
                balance_df: pd.DataFrame, checks: list | None = None) -> str:
    """生成可归档的纯文本评审报告。"""
    lines = []
    lines.append("=" * 72)
    lines.append("汽改电评审：全厂公用工程边界联合归因报告")
    lines.append("=" * 72)
    lines.append(f"价格/排放情景：{scenario}    证据时点 as_of：{pack.as_of or '全部资料（事后复核）'}")
    lines.append(f"基线日窗：改造前非检修 {diag['baseline_days']} 天；"
                 f"基线资料本身误差界外冲突 {diag['baseline_conflict_days']} 天")
    if diag.get("baseline_mean_slack_t", 0) > 0:
        lines.append(f"基线日均节点松弛 {diag['baseline_mean_slack_t']:.3f} t（资料测量能力之外的不平衡）")
    lines.append("")
    cm = diag["coal_model"]
    lines.append("[煤耗关系] " + cm.note)
    lines.append("")
    lines.append("[方案能源、费用与排放变化，每日均值；费用/排放正=增加]")
    lines.append(money_table(result).to_string(index=False))
    lines.append("")
    if not diag["interaction"].empty:
        lines.append("[联合交互] 组合 AB 与单项 A+B 的区间差异")
        lines.append(diag["interaction"].round(2).to_string(index=False))
    lines.append("")
    n_conflict = int((balance_df.status == "conflict").sum())
    lines.append(f"[平衡可行性] 60 个日窗中 {n_conflict} 天存在仪表误差无法解释的节点不平衡；")
    lines.append("冲突日的流量推断带显式松弛，未计量边只报容量内可行区间，不报确定流量。")
    if checks:
        lines.append("")
        lines.append("[改造后承诺检验]")
        for c in checks:
            lines.append(
                f"  - {c.item}：承诺 {c.committed}，观测 {fmt_interval(c.observed_lo, c.observed_hi)}"
                f" {c.unit}，结论 {c.verdict}；{c.detail}"
            )
    lines.append("")
    lines.append("口径声明：减汽×固定单价不构成厂区节约；煤项在煤-汽关系不可识别时记 0 并保留范围；")
    lines.append("所有区间端点可由相同资料与 as_of 切片重现。")
    return "\n".join(lines)
