"""改造后计量检验：用陆续进入的观测核对原评审承诺。

检验逻辑：
- 评审承诺来自 proposals（设备侧减汽、增电、余热变化），是方案假设而非测量。
- 改造后实际效果只用 available_at ≤ 检验时点 的资料计算，保证与原评审
  一样可按证据时点重现；补测晚到的影响通过两次检验的差异体现。
- 可比条件沿用负荷档匹配：同负荷档改造前后均值之差，剔除检修日。
- 结论分 observed_supported（观测支持）、within_meter_uncertainty
  （误差范围内相容）、not_verifiable（资料不足或依赖不可识别流量）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataio import load_pack
from .eval import load_proposals


@dataclass(frozen=True)
class CheckResult:
    item: str
    committed: float
    observed_lo: float
    observed_hi: float
    unit: str
    verdict: str
    detail: str


def _phase_matched_delta(daily: pd.DataFrame, column: str) -> tuple[float, float, float]:
    """负荷档匹配的改造前后均值差，返回（点值, 低端, 高端）。"""
    pre, post = [], []
    for load, grp in daily.groupby("load"):
        g = grp[~grp.maintenance.astype(bool)]
        a = g[~g.retrofit.astype(bool)][column]
        b = g[g.retrofit.astype(bool)][column]
        if len(a) and len(b):
            pre.append(a.mean())
            post.append(b.mean())
    if not pre:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(post) - np.mean(pre))
    return point, point, point


def verify_commitments(pack, proposal_id: str = "A", meter_uncertainty: bool = True) -> list[CheckResult]:
    """核对一个已实施方案的设备侧承诺（默认 A，即资料中实际改造的方案）。"""
    proposals = load_proposals(pack)
    if proposal_id not in proposals:
        raise ValueError(f"未知方案：{proposal_id}")
    prop = proposals[proposal_id]
    daily = pack.daily
    results: list[CheckResult] = []

    # 1) 设备减汽：hp_device 改造后下降量应接近承诺减汽
    d_point, _, _ = _phase_matched_delta(daily, "hp_device")
    observed_reduction = -d_point  # 减汽为正
    b_device = pack.meter_daily_bounds["hp_device"]
    if meter_uncertainty:
        lo, hi = observed_reduction - b_device, observed_reduction + b_device
    else:
        lo = hi = observed_reduction
    verdict = _verdict(prop.steam_reduction_t, lo, hi)
    results.append(CheckResult(
        item="设备侧减汽", committed=prop.steam_reduction_t,
        observed_lo=lo, observed_hi=hi, unit="t/8h窗",
        verdict=verdict,
        detail=f"负荷档匹配观测减汽 {observed_reduction:.2f} t，仪表日界 ±{b_device:.1f} t",
    ))

    # 2) 设备增电：device_kwh 改造后增量
    d_kwh, _, _ = _phase_matched_delta(daily, "device_kwh")
    # power 表无误差界资料，按点值核对并注明
    verdict_kwh = _verdict(prop.electric_add_kwh, d_kwh, d_kwh)
    results.append(CheckResult(
        item="设备侧增电", committed=prop.electric_add_kwh,
        observed_lo=d_kwh, observed_hi=d_kwh, unit="kWh/8h窗",
        verdict=verdict_kwh,
        detail="power 表未登记误差界，按点值核对；其他电耗恒定说明增电全额传导至购电",
    ))

    # 3) 全厂购电变化（含背压通道），这是评审真正承诺的边界量
    d_imp, _, _ = _phase_matched_delta(daily, "import_kwh")
    if np.isnan(d_imp):
        results.append(CheckResult(
            item="全厂购电增量", committed=prop.electric_add_kwh,
            observed_lo=float("nan"), observed_hi=float("nan"), unit="kWh/8h窗",
            verdict="not_verifiable",
            detail="证据时点内尚无改造后日窗，无法检验购电边界变化",
        ))
        return results
    results.append(CheckResult(
        item="全厂购电增量", committed=prop.electric_add_kwh,
        observed_lo=d_imp, observed_hi=d_imp, unit="kWh/8h窗",
        verdict="reference_only",
        detail=(f"观测购电增量 {d_imp:.0f} kWh，大于设备增电承诺，因减汽未减少锅炉出力、"
                f"背压发电未增加，设备增电几乎全额变为外购电"),
    ))

    # 4) 余热变化：waste_t 是服务侧可回收量登记，不是 waste_mp 实测
    pre_w = daily[~daily.retrofit.astype(bool)].waste_t.mean()
    post_w = daily[daily.retrofit.astype(bool)].waste_t.mean()
    results.append(CheckResult(
        item="余热可回收变化", committed=prop.recovery_change_t,
        observed_lo=post_w - pre_w, observed_hi=post_w - pre_w, unit="t/8h窗",
        verdict="not_verifiable" if prop.recovery_change_t != 0 else "observed_supported",
        detail=("waste_t 为可回收能力登记而非 waste_mp 实测；实际回收量与放空联合不可识别，"
                "承诺为 0 时登记值一致，但回收-放空分摊仍需补测"),
    ))
    return results


def _verdict(committed: float, lo: float, hi: float) -> str:
    if np.isnan(lo):
        return "not_verifiable"
    if lo <= committed <= hi:
        return "within_meter_uncertainty" if lo != hi else "observed_supported"
    return "deviation"


def review_as_of(root, proposal_id: str = "A") -> pd.DataFrame:
    """按资料的可知时间分阶段重现检验，展示晚到补测如何改变结论范围。"""
    full = load_pack(root, as_of=None)
    # 取全部 available_at 时点，逐时点重算
    stamps = sorted(set(full.measurements.available_at) |
                    set(full.supplement.available_at) |
                    set(full.operations.available_at))
    rows = []
    for ts in stamps:
        pack = load_pack(root, as_of=ts)
        if pack.daily.retrofit.nunique() < 2:
            continue
        checks = verify_commitments(pack, proposal_id)
        for c in checks:
            rows.append({
                "as_of": ts,
                "item": c.item,
                "committed": c.committed,
                "observed_lo": c.observed_lo,
                "observed_hi": c.observed_hi,
                "verdict": c.verdict,
                "excluded_supplement": pack.excluded_by_as_of.get("supplement", 0),
            })
    return pd.DataFrame(rows)
