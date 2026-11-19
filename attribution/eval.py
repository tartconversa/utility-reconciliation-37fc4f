"""单项与组合方案的可比工况反事实评估与联合归因。

方法纪律：
- 基线取改造前、非检修日的负荷匹配工况，服务量分布与改造后一致，
  构造“相同服务量”的可比条件；检修日不进入基线。
- 设备侧假设（减汽、增电、余热变化）只作为 hp_device/device_kwh/waste 容量
  的反事实边界输入，全厂购能变化由网络在物理容量与误差界内重新平衡得到，
  不把“减汽 × 固定单价”直接记为节约。
- 对每个输出求 min/max 可行区间：依赖不可识别流量（mp_vent/waste_mp）的
  结论给区间并标注，不报确定性点值。
- 计量边带显式松弛：基线日本身若超出仪表误差能力，冲突量被记录并惩罚，
  不用反推未观测流量把冲突“平衡掉”。
- 组合 AB 在同一 LP 中整体求解，再与单项结果对比，显式暴露余热交互，
  不假设单项收益可加。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from .network import EDGE_ORDER, NODE_BALANCES, Network

_SLACK_PENALTY = 1_000.0
_METERED = ["boiler_hp", "hp_mp", "mp_lp", "hp_device", "lp_user"]
_UNMETERED = ["waste_mp", "mp_vent", "lp_loss"]
_NODES = list(NODE_BALANCES.keys())


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    compressor_count: int
    steam_reduction_t: float
    electric_add_kwh: float
    recovery_change_t: float


def load_proposals(pack) -> dict[str, Proposal]:
    return {
        r.proposal_id: Proposal(
            proposal_id=r.proposal_id,
            compressor_count=int(r.compressor_count),
            steam_reduction_t=float(r.steam_reduction_t),
            electric_add_kwh=float(r.electric_add_kwh),
            recovery_change_t=float(r.recovery_change_t),
        )
        for r in pack.proposals.itertuples()
    }


def baseline_window(pack) -> pd.DataFrame:
    """评审基线日窗：改造前且非检修。"""
    d = pack.daily
    return d[(~d.retrofit.astype(bool)) & (~d.maintenance.astype(bool))].copy()


class _Counterfactual:
    """单个基线日窗在给定方案下的 LP。

    变量：x[0:8] 八条边流量；x[8:14] 三个节点的正/负松弛（非负，高惩罚）。
    计量边流量限定在“目标中心 ± 仪表日界”内，与物理容量取交集——界内移动
    是测量允许的重新平衡，不算冲突；只有任何界内取值都无法满足节点平衡时，
    松弛才非零，其总量即该日窗真正无法由误差解释的冲突。
    点解目标取流量到目标中心的归一化 L1 距离，可行时自然落在中心。
    """

    def __init__(self, net: Network, base: dict[str, float], meter_bounds: dict[str, float],
                 waste_cap_base: float, proposal: Proposal | None):
        self.net = net
        self.base = base
        self.bounds_map = meter_bounds
        self.proposal = proposal
        self.n_edge = len(EDGE_ORDER)
        self.s0 = self.n_edge                       # 松弛变量起点
        self.n_var = self.s0 + 2 * len(_NODES)
        self.waste_cap = max(
            0.0,
            min(waste_cap_base + (proposal.recovery_change_t if proposal else 0.0),
                net.bounds("waste_mp")[1]),
        )
        self.centers = self._centers()
        self._build()

    def _centers(self) -> dict[str, float]:
        centers = {e: self.base[e] for e in _METERED}
        if self.proposal is not None:
            centers["hp_device"] -= self.proposal.steam_reduction_t
        return centers

    def _build(self) -> None:
        idx = self.net.index
        lb = np.full(self.n_var, -np.inf)
        ub = np.full(self.n_var, np.inf)

        # 计量边：目标中心 ± 误差界，与物理容量取交集
        for e in _METERED:
            cap_lo, cap_hi = self.net.bounds(e)
            bound = self.bounds_map[e]
            lb[idx[e]] = max(cap_lo, self.centers[e] - bound)
            ub[idx[e]] = min(cap_hi, self.centers[e] + bound)
        # 未计量边：只有物理容量；waste_mp 另受可回收量上限
        for e in _UNMETERED:
            cap_lo, cap_hi = self.net.bounds(e)
            lb[idx[e]], ub[idx[e]] = cap_lo, cap_hi
        ub[idx["waste_mp"]] = min(ub[idx["waste_mp"]], self.waste_cap)
        # 节点松弛非负
        for k in range(self.s0, self.n_var):
            lb[k], ub[k] = 0.0, np.inf
        self.lb, self.ub = lb, ub

        rows, cols, data = [], [], []
        for ni, node in enumerate(_NODES):
            for e, coef in NODE_BALANCES[node].items():
                rows.append(ni); cols.append(idx[e]); data.append(coef)
            rows.append(ni); cols.append(self.s0 + 2 * ni); data.append(1.0)
            rows.append(ni); cols.append(self.s0 + 2 * ni + 1); data.append(-1.0)
        self.A_eq = csr_matrix(
            (data, (rows, cols)), shape=(len(_NODES), self.n_var))
        self.b_eq = np.zeros(len(_NODES))

    def _solve(self, c: np.ndarray) -> np.ndarray:
        res = linprog(c, A_eq=self.A_eq, b_eq=self.b_eq,
                      bounds=list(zip(self.lb, self.ub)), method="highs")
        if res.status != 0:
            raise RuntimeError(f"反事实 LP 失败 status={res.status} {res.message}")
        return res.x

    def point(self) -> tuple[np.ndarray, float]:
        """到目标中心归一化 L1 距离最小的基准解与节点总松弛。"""
        c = np.zeros(self.n_var)
        # L1 距离需要辅助变量：再扩展 2*|metered| 个偏差变量
        m = len(_METERED)
        ext = self.n_var + 2 * m
        lb2 = np.concatenate([self.lb, np.zeros(2 * m)])
        ub2 = np.concatenate([self.ub, np.full(2 * m, np.inf)])
        c2 = np.zeros(ext)
        ar, ac, ad = [], [], []
        for k, e in enumerate(_METERED):
            # x_edge - u+ + u- = center
            ar += [k, k, k]
            ac += [self.net.index[e], self.n_var + 2 * k, self.n_var + 2 * k + 1]
            ad += [1.0, -1.0, 1.0]
            c2[self.n_var + 2 * k] = 1.0 / max(self.bounds_map[e], 1e-9)
            c2[self.n_var + 2 * k + 1] = 1.0 / max(self.bounds_map[e], 1e-9)
        for ni in range(len(_NODES)):
            c2[self.s0 + 2 * ni] = _SLACK_PENALTY
            c2[self.s0 + 2 * ni + 1] = _SLACK_PENALTY
        A_dev = csr_matrix((ad, (ar, ac)), shape=(m, ext))
        n_node = len(_NODES)
        A_node = csr_matrix(
            (self.A_eq.data, self.A_eq.nonzero()), shape=(n_node, ext))
        big = csr_matrix(
            (np.concatenate([A_node.data, A_dev.data]),
             (np.concatenate([A_node.nonzero()[0], A_dev.nonzero()[0] + n_node]),
              np.concatenate([A_node.nonzero()[1], A_dev.nonzero()[1]]))),
            shape=(n_node + m, ext),
        )
        b2 = np.concatenate([np.zeros(n_node), np.array([self.centers[e] for e in _METERED])])
        res = linprog(c2, A_eq=big, b_eq=b2, bounds=list(zip(lb2, ub2)), method="highs")
        if res.status != 0:
            raise RuntimeError(f"反事实基准 LP 失败 status={res.status} {res.message}")
        x = res.x[: self.n_var]
        slack = float(x[self.s0:].sum())
        return x, slack

    def edge_extreme(self, edge: str, minimize: bool, slack_cap: float) -> np.ndarray:
        """固定在最小松弛面上求某边流量的 min/max。"""
        saved_ub = self.ub.copy()
        for ni in range(len(_NODES)):
            self.ub[self.s0 + 2 * ni] = min(self.ub[self.s0 + 2 * ni], slack_cap + 1e-6)
            self.ub[self.s0 + 2 * ni + 1] = min(self.ub[self.s0 + 2 * ni + 1], slack_cap + 1e-6)
        c = np.zeros(self.n_var)
        c[self.net.index[edge]] = 1.0 if minimize else -1.0
        try:
            x = self._solve(c)
        finally:
            self.ub = saved_ub
        return x


def evaluate_proposal(net: Network, pack, proposal: Proposal | None) -> dict:
    """对一个方案（None=基线）在全部基线日窗上求各边流量的可行区间。

    conflict_days 始终报告该日窗在对应设定下的最小节点松弛量；
    方案与基线的冲突日数之差即“方案诱发冲突”，由 run_review 比较。
    """
    base_days = baseline_window(pack)
    bounds = pack.meter_daily_bounds
    per_day = {e: {"lo": [], "hi": [], "point": []} for e in EDGE_ORDER}
    slack_total = 0.0
    conflict_days = 0
    for row in base_days.itertuples():
        base = {e: float(getattr(row, e)) for e in _METERED}
        cf = _Counterfactual(net, base, bounds, float(row.waste_t), proposal)
        x0, slack = cf.point()
        slack_total += slack
        if slack > 1e-5:
            conflict_days += 1
        for e in EDGE_ORDER:
            lo_x = cf.edge_extreme(e, True, slack)
            hi_x = cf.edge_extreme(e, False, slack)
            per_day[e]["lo"].append(float(lo_x[net.index[e]]))
            per_day[e]["hi"].append(float(hi_x[net.index[e]]))
            per_day[e]["point"].append(float(x0[net.index[e]]))
    summary = {
        e: {k: float(np.mean(v[k])) for k in ("lo", "hi", "point")}
        for e, v in per_day.items()
    }
    return {
        "proposal": proposal,
        "n_days": len(base_days),
        # 基线设定下冲突=资料本身不可由误差解释；方案设定下额外的冲突=
        # 反事实减汽在该日窗测量界内找不到物理平衡（反事实不可平衡日）。
        "conflict_days": conflict_days,
        "mean_slack_t": slack_total / max(len(base_days), 1),
        "flows": summary,
        "bp_kwh_per_t": float(base_days.backpressure_kwh_t.iloc[0]),
    }


def energy_and_money(
    base_eval: dict,
    cf_eval: dict,
    proposal: Proposal,
    external: pd.Series,
    coal_identified: bool,
    coal_steam_coef: float,
) -> dict:
    """把网络反事实区间转换为全厂能源、费用与排放变化（每日均值）。

    电力：Δ购电 = 设备增电 - Δ背压发电；背压发电 = hp_mp × 系数。
    煤：仅当煤-汽偏效应可识别时才计算 Δ煤；否则记 0 并标注“不可识别”，
        观测到的阶段煤耗差由调用方作为未归因对照单独展示，不并入项目收益。
    """
    bf, cf = base_eval["flows"], cf_eval["flows"]
    bp = base_eval["bp_kwh_per_t"]

    delta_hp_mp_lo = cf["hp_mp"]["lo"] - bf["hp_mp"]["hi"]
    delta_hp_mp_hi = cf["hp_mp"]["hi"] - bf["hp_mp"]["lo"]
    delta_bp_lo, delta_bp_hi = delta_hp_mp_lo * bp, delta_hp_mp_hi * bp

    d_import_lo = proposal.electric_add_kwh - delta_bp_hi
    d_import_hi = proposal.electric_add_kwh - delta_bp_lo

    d_boiler_lo = cf["boiler_hp"]["lo"] - bf["boiler_hp"]["hi"]
    d_boiler_hi = cf["boiler_hp"]["hi"] - bf["boiler_hp"]["lo"]

    if coal_identified:
        d_coal_lo = d_boiler_lo * coal_steam_coef
        d_coal_hi = d_boiler_hi * coal_steam_coef
        coal_status = "identified"
    else:
        d_coal_lo = d_coal_hi = 0.0
        coal_status = "not_identifiable"

    p_pwr, p_coal = float(external.power_yuan_kwh), float(external.coal_yuan_t)
    e_pwr, e_coal = float(external.power_kg_kwh), float(external.coal_kg_t)

    cost_lo = d_import_lo * p_pwr + min(d_coal_lo, d_coal_hi) * p_coal
    cost_hi = d_import_hi * p_pwr + max(d_coal_lo, d_coal_hi) * p_coal
    emis_lo = d_import_lo * e_pwr + min(d_coal_lo, d_coal_hi) * e_coal
    emis_hi = d_import_hi * e_pwr + max(d_coal_lo, d_coal_hi) * e_coal

    return {
        "proposal_id": proposal.proposal_id,
        "scenario": external.scenario,
        "conflict_days": cf_eval["conflict_days"],
        "mean_slack_t": cf_eval["mean_slack_t"],
        "delta_boiler_t": (d_boiler_lo, d_boiler_hi),
        "delta_hp_mp_t": (delta_hp_mp_lo, delta_hp_mp_hi),
        "delta_backpressure_kwh": (delta_bp_lo, delta_bp_hi),
        "device_add_kwh": proposal.electric_add_kwh,
        "delta_import_kwh": (d_import_lo, d_import_hi),
        "delta_coal_t": (d_coal_lo, d_coal_hi),
        "coal_status": coal_status,
        "delta_cost_yuan": (cost_lo, cost_hi),
        "delta_emission_kg": (emis_lo, emis_hi),
    }


def _flatten(money: dict) -> dict:
    out = {
        "proposal_id": money["proposal_id"],
        "scenario": money["scenario"],
        "coal_status": money["coal_status"],
        "conflict_days": money["conflict_days"],
        "induced_conflict_days": money.get("induced_conflict_days", 0),
        "mean_slack_t": round(money["mean_slack_t"], 4),
        "device_add_kwh": money["device_add_kwh"],
    }
    for k, v in money.items():
        if isinstance(v, tuple):
            out[k + "_lo"] = v[0]
            out[k + "_hi"] = v[1]
    return out


def _interaction_terms(evals, proposals, external, coal_ok, coal_coef) -> pd.DataFrame:
    """组合 AB 相对 A+B 的交互量（基于区间端点的保守比较）。"""
    if not {"A", "B", "AB"}.issubset(evals):
        return pd.DataFrame()
    base = evals["baseline"]
    m = {
        pid: energy_and_money(base, evals[pid], prop, external, coal_ok, coal_coef)
        for pid, prop in proposals.items()
    }
    rows = []
    for metric in ["delta_cost_yuan", "delta_emission_kg", "delta_import_kwh", "delta_boiler_t"]:
        ab_lo, ab_hi = m["AB"][metric]
        sum_lo = m["A"][metric][0] + m["B"][metric][0]
        sum_hi = m["A"][metric][1] + m["B"][metric][1]
        inter_lo, inter_hi = ab_lo - sum_hi, ab_hi - sum_lo
        rows.append({
            "metric": metric,
            "AB_lo": ab_lo, "AB_hi": ab_hi,
            "AplusB_lo": sum_lo, "AplusB_hi": sum_hi,
            "interaction_lo": inter_lo,
            "interaction_hi": inter_hi,
            "contains_zero": inter_lo <= 0.0 <= inter_hi,
            "note": "区间含 0：现有测量无法证明交互，单项加总与组合不相悖"
                    if inter_lo <= 0.0 <= inter_hi
                    else "区间不含 0：单项变化不可直接相加（余热回收耦合已被资料识别）",
        })
    return pd.DataFrame(rows)


def run_review(pack, scenario: str = "central") -> tuple[pd.DataFrame, dict]:
    """完整评审：基线 + 全部单项与组合，在同一价格/排放假设下输出可比结果。"""
    from .network import build_network
    from .models import fit_coal_model

    net = build_network(pack.edges)
    proposals = load_proposals(pack)
    if scenario not in set(pack.external.scenario):
        raise ValueError(
            f"未知外部情景：{scenario}；可选 {sorted(pack.external.scenario)}")
    external = pack.external[pack.external.scenario == scenario].iloc[0]
    coal_model = fit_coal_model(pack.daily)
    coal_coef = coal_model.coefficients.get("boiler_hp_c", 0.0)

    base_eval = evaluate_proposal(net, pack, None)
    rows, evals = [], {"baseline": base_eval}
    for pid, prop in proposals.items():
        cf = evaluate_proposal(net, pack, prop)
        evals[pid] = cf
        money = energy_and_money(base_eval, cf, prop, external,
                                 coal_model.identifiable, coal_coef)
        # 方案诱发冲突 = 该设定冲突日数 - 基线固有冲突日数
        money["induced_conflict_days"] = cf["conflict_days"] - base_eval["conflict_days"]
        rows.append(_flatten(money))
    result = pd.DataFrame(rows)
    interaction = _interaction_terms(evals, proposals, external,
                                     coal_model.identifiable, coal_coef)
    diag = {
        "coal_model": coal_model,
        "baseline_days": base_eval["n_days"],
        "baseline_conflict_days": base_eval["conflict_days"],
        "baseline_mean_slack_t": base_eval["mean_slack_t"],
        "interaction": interaction,
        "scenario": scenario,
        "as_of": pack.as_of,
        "evals": evals,
    }
    return result, diag
