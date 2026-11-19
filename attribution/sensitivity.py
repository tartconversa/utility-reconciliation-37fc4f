"""敏感性与决策支持：排序稳健性、测量收紧、测量/价格变化来源分解。

回答评审的最后一公里问题：
- 组合排序在区间重叠时是否稳健？
- 仪表误差收紧或补测进入后，可行平衡、归因和排序如何变化？
- 两次评审结果的差异来自测量改进还是外部价格假设？
- 哪些收益项已有测量支持、哪些仍需补测才能进入投资决策？
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from . import dataio, eval as eval_mod
from .dataio import EDGE_METERS


def with_meter_bounds(pack, meter_bounds_override: dict[str, float]):
    """用新的单读数绝对误差界构造资料包副本，不触碰原始观测。

    键为 edge_id（boiler_hp 等），值为该仪表“单条读数”的新界；
    日窗界按每日读数条数重新相加，与 dataio 的口径一致。
    """
    spec = pack.meter_specs.copy()
    new_daily: dict[str, float] = {}
    counts = (pack.measurements.groupby("meter_id").size() /
              pack.daily.shape[0]).astype(int)
    for edge_id, bound in meter_bounds_override.items():
        meter_id = EDGE_METERS[edge_id]
        spec.loc[spec.meter_id == meter_id, "bound"] = bound
        new_daily[edge_id] = float(bound) * int(counts[meter_id])
    merged = {**pack.meter_daily_bounds, **new_daily}
    return replace(pack, meter_specs=spec, meter_daily_bounds=merged)


def rank_proposals(result: pd.DataFrame, metric: str = "delta_cost_yuan") -> pd.DataFrame:
    """按费用/排放区间排序，并标注区间重叠导致的排序不确定性。

    本资料下所有方案都是净增支出，“最优”=增量最小；区间不重叠的排序
    才是稳健排序，重叠时标注 needs_review。
    """
    rows = []
    for _, r in result.iterrows():
        rows.append({
            "proposal_id": r.proposal_id,
            "scenario": r.scenario,
            "lo": r[f"{metric}_lo"],
            "hi": r[f"{metric}_hi"],
            "coal_status": r.coal_status,
        })
    out = pd.DataFrame(rows).sort_values("lo").reset_index(drop=True)
    robust = [True]
    for i in range(1, len(out)):
        # 与前一名的区间不重叠才算稳健超越
        robust.append(out.loc[i, "lo"] >= out.loc[i - 1, "hi"])
    out["rank"] = range(1, len(out) + 1)
    out["ranking_robust"] = robust
    out["ranking_note"] = [
        "区间不重叠，排序稳健" if ok else "与更优方案区间重叠，排序依赖不可识别流量，需补测"
        for ok in robust
    ]
    return out


def measurement_vs_price_decomposition(pack) -> pd.DataFrame:
    """分解同一方案在不同情景间的结果差异来源。

    物理量（购电、煤、汽）不随 external 情景变化；变化的只能是价格/排放因子。
    返回每个方案 × 情景对的费用差，并显式标注来源为外部价格而非测量。
    """
    sweep = []
    for sc in pack.external.scenario:
        result, _ = eval_mod.run_review(pack, sc)
        sweep.append(result)
    full = pd.concat(sweep, ignore_index=True)
    base_sc = pack.external.scenario.iloc[0]
    base = full[full.scenario == base_sc].set_index("proposal_id")
    rows = []
    for _, r in full[full.scenario != base_sc].iterrows():
        b = base.loc[r.proposal_id]
        rows.append({
            "proposal_id": r.proposal_id,
            "comparison": f"{base_sc} -> {r.scenario}",
            "physical_change": "无（购电/煤/汽区间完全相同，差异不来自测量）",
            "cost_delta_vs_base_lo": r.delta_cost_yuan_lo - b.delta_cost_yuan_lo,
            "cost_delta_vs_base_hi": r.delta_cost_yuan_hi - b.delta_cost_yuan_hi,
            "emission_delta_vs_base_lo": r.delta_emission_kg_lo - b.delta_emission_kg_lo,
            "emission_delta_vs_base_hi": r.delta_emission_kg_hi - b.delta_emission_kg_hi,
            "source": "外部价格/排放因子",
        })
    return pd.DataFrame(rows)


def decision_summary(pack, scenario: str = "central") -> pd.DataFrame:
    """投资决策视图：逐项列出收益/成本项的证据状态。

    supported      有可计量观测或可识别关系支持，可进入决策
    range_only     只能给区间，决策需接受保守端点
    needs_measure  依赖不可识别流量或不可识别关系，必须补测后才能定论
    """
    result, diag = eval_mod.run_review(pack, scenario)
    rows = []
    for _, r in result.iterrows():
        rows += [
            {"proposal_id": r.proposal_id, "item": "设备增电与购电增量",
             "status": "supported",
             "evidence": "power 表直接计量，电力平衡中其他电耗对阶段不敏感",
             "value": f"购电增量 [{r.delta_import_kwh_lo:.0f}, {r.delta_import_kwh_hi:.0f}] kWh/日"},
            {"proposal_id": r.proposal_id, "item": "购电费用与排放",
             "status": "range_only",
             "evidence": "背压汽量区间导致购电区间；价格/因子为外部线性假设",
             "value": f"费用 [{r.delta_cost_yuan_lo:.0f}, {r.delta_cost_yuan_hi:.0f}] 元/日"},
            {"proposal_id": r.proposal_id, "item": "节煤收益",
             "status": "needs_measure",
             "evidence": "煤-汽偏效应不可识别（控制星期/阶段后 ΔR²=0），减汽未传导到锅炉",
             "value": "记 0，不得把减汽×单价计为节约；需补测锅炉煤-汽关系或煤表分炉计量"},
            {"proposal_id": r.proposal_id, "item": "余热回收/放空分摊",
             "status": "needs_measure",
             "evidence": "waste_mp 与 mp_vent 联合不可识别，仅 2 天有补测",
             "value": "需连续放空/回收计量后才能量化组合方案的余热交互"},
            {"proposal_id": r.proposal_id, "item": "反事实可平衡性",
             "status": ("range_only" if r.induced_conflict_days == 0 else "needs_measure"),
             "evidence": f"{r.induced_conflict_days} 个基线日窗在减汽后超出仪表误差可平衡能力",
             "value": f"日均 HP 松弛 {r.mean_slack_t:.2f} t；收紧 M1/M2/M4 误差界可消除"},
        ]
    return pd.DataFrame(rows)
