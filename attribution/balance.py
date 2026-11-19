"""误差界内的可行平衡（线性规划）。

原则（对应评审要求）：
- 可信测量不得任意修改：计量边流量只能在观测 ± bound 内取值，目标优先零调整。
- 未计量边只有容量范围，无已知流量；不允许隐含平衡插件。
- 初始观测不保证彼此可行：无可行解时报告“真正冲突”及最小松弛量，
  不把未观测流量反推成确定事实。
- 输出严格区分原始观测（observed）、可行推断（estimate）和可行区间（range）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from .network import EDGE_ORDER, METERED_EDGES, NODE_BALANCES, Network

# 松弛变量的惩罚量级：远大于误差界归一化后的调整成本，
# 保证“能不破坏平衡就不破坏”，但冲突时仍给出最小冲突解而非无解。
SLACK_PENALTY = 1_000.0


@dataclass(frozen=True)
class DayBalance:
    day: pd.Timestamp
    status: str                 # feasible（界内可行）/ conflict（界外冲突，用了松弛）
    observed: dict[str, float]  # 原始日窗观测，未计量边为 NaN
    estimate: dict[str, float]  # LP 最小调整解
    lower: dict[str, float]     # 最小调整前提下的可行区间
    upper: dict[str, float]
    deviations: dict[str, float]
    slack: dict[str, float]     # 各节点超出测量能力的最小不平衡量（吨）
    note: str


def _variable_layout(net: Network):
    """变量布局：x = [8 条边流量, 5 个计量偏差 d, 3 节点正负松弛 s]。"""
    n_edge = len(EDGE_ORDER)
    meter_pos = {e: i for i, e in enumerate(METERED_EDGES)}
    d0 = n_edge
    s0 = d0 + len(METERED_EDGES)
    nodes = list(NODE_BALANCES.keys())
    return n_edge, d0, s0, nodes, meter_pos


def _bounds_and_obs(net: Network, observed: pd.Series, meter_bounds: dict[str, float],
                    waste_cap_t: float, vent_supplement: tuple[float, float] | None):
    """构造变量上下界与观测向量。"""
    n_edge, d0, s0, nodes, meter_pos = _variable_layout(net)
    n_var = s0 + 2 * len(nodes)
    lb = np.full(n_var, -np.inf)
    ub = np.full(n_var, np.inf)
    obs = {e: np.nan for e in EDGE_ORDER}

    idx = net.index
    for e in EDGE_ORDER:
        lo, hi = net.bounds(e)
        j = idx[e]
        if e == "waste_mp":
            hi = min(hi, float(waste_cap_t))   # 实际回收不超过可回收汽量 waste_t
        if e in METERED_EDGES:
            value = float(observed[e])
            bound = float(meter_bounds[e])
            obs[e] = value
            lb[j], ub[j] = value - bound, value + bound
            k = d0 + meter_pos[e]
            lb[k], ub[k] = 0.0, bound          # 偏差不得超过仪表误差界
        else:
            lb[j], ub[j] = lo, hi
    # LP 松弛变量非负
    for k in range(s0, n_var):
        lb[k], ub[k] = 0.0, np.inf
    if vent_supplement is not None:
        v_value, v_bound = vent_supplement
        j = idx["mp_vent"]
        lb[j] = max(lb[j], v_value - v_bound)
        ub[j] = min(ub[j], v_value + v_bound)
    return lb, ub, obs


def _balance_constraints(net: Network):
    """节点平衡等式，松弛项允许在测量能力之外显式不平衡。"""
    n_edge, d0, s0, nodes, meter_pos = _variable_layout(net)
    n_var = s0 + 2 * len(nodes)
    rows, cols, data, rhs = [], [], [], []
    for ni, node in enumerate(nodes):
        for e, coef in NODE_BALANCES[node].items():
            rows.append(ni); cols.append(net.index[e]); data.append(coef)
        rows.append(ni); cols.append(s0 + 2 * ni); data.append(+1.0)   # 正松弛
        rows.append(ni); cols.append(s0 + 2 * ni + 1); data.append(-1.0)
        rhs.append(0.0)
    return csr_matrix((data, (rows, cols)), shape=(len(nodes), n_var)), np.array(rhs), nodes


def _solve(net: Network, observed: pd.Series, meter_bounds: dict[str, float],
           waste_cap_t: float, vent_supplement, direction: str = "min",
           objective_edge: str | None = None, slack_cap: float | None = None):
    """统一 LP 求解。

    direction='min_deviation'：最小化归一化仪表调整 + 高惩罚松弛（基准解）。
    direction='min'/'max'：在 slack_cap 限定的最小松弛面上求 objective_edge
    的区间端点，避免用不可解释的额外不平衡换取更宽区间。
    """
    n_edge, d0, s0, nodes, meter_pos = _variable_layout(net)
    n_var = s0 + 2 * len(nodes)
    A_eq, b_eq, nodes = _balance_constraints(net)
    lb, ub, obs = _bounds_and_obs(net, observed, meter_bounds, waste_cap_t, vent_supplement)

    # 区间 LP 固定在基准解找到的最小松弛水平（加 1e-6 数值容差）。
    if slack_cap is not None:
        for ni in range(len(nodes)):
            ub[s0 + 2 * ni] = min(ub[s0 + 2 * ni], slack_cap + 1e-6)
            ub[s0 + 2 * ni + 1] = min(ub[s0 + 2 * ni + 1], slack_cap + 1e-6)

    c = np.zeros(n_var)
    if direction == "min_deviation":
        for e, pos in meter_pos.items():
            c[d0 + pos] = 1.0 / max(meter_bounds[e], 1e-9)
        for ni in range(len(nodes)):
            c[s0 + 2 * ni] = SLACK_PENALTY
            c[s0 + 2 * ni + 1] = SLACK_PENALTY
    else:
        c[net.index[objective_edge]] = 1.0 if direction == "min" else -1.0

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lb, ub)), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP 求解失败 status={res.status} {res.message}")
    return res.x, nodes, s0, obs


def balance_day(net: Network, observed: pd.Series, meter_bounds: dict[str, float],
                waste_cap_t: float, vent_supplement: tuple[float, float] | None = None) -> DayBalance:
    """对单个 8 小时日窗求可行平衡、冲突诊断与未计量流量可行区间。"""
    x, nodes, s0, obs = _solve(net, observed, meter_bounds, waste_cap_t,
                               vent_supplement, "min_deviation")
    idx = net.index
    n_edge, d0, _, _, meter_pos = _variable_layout(net)

    slack = {}
    total_slack = 0.0
    for ni, node in enumerate(nodes):
        s = float(x[s0 + 2 * ni] + x[s0 + 2 * ni + 1])
        slack[node] = s
        total_slack += s
    status = "conflict" if total_slack > 1e-5 else "feasible"

    estimate = {e: float(x[idx[e]]) for e in EDGE_ORDER}
    deviations = {e: float(x[d0 + meter_pos[e]]) for e in METERED_EDGES}

    # 区间：计量边直接给观测±界；未计量边在“不增加松弛”的最小冲突面上求 min/max。
    lower, upper = {}, {}
    for e in METERED_EDGES:
        lower[e] = obs[e] - meter_bounds[e]
        upper[e] = obs[e] + meter_bounds[e]
    for e in ("waste_mp", "mp_vent", "lp_loss"):
        lo_x, _, _, _ = _solve(net, observed, meter_bounds, waste_cap_t,
                               vent_supplement, "min", e, slack_cap=total_slack)
        hi_x, _, _, _ = _solve(net, observed, meter_bounds, waste_cap_t,
                               vent_supplement, "max", e, slack_cap=total_slack)
        lower[e] = float(lo_x[idx[e]])
        upper[e] = float(hi_x[idx[e]])

    note = "误差界内可平衡" if status == "feasible" else \
        f"测量误差无法解释的节点不平衡：" + ",".join(f"{n}={v:.2f}t" for n, v in slack.items() if v > 1e-5)
    return DayBalance(
        day=pd.Timestamp(observed["day"]),
        status=status,
        observed=obs,
        estimate=estimate,
        lower=lower,
        upper=upper,
        deviations=deviations,
        slack=slack,
        note=note,
    )


def balance_pack(pack) -> pd.DataFrame:
    """对资料包内全部日窗逐天平衡，返回逐日结果表（观测/推断/区间分列）。"""
    net = build_network_from_pack(pack)
    vent_lookup = {
        row.day: (float(row.value), float(row.bound))
        for row in pack.supplement.itertuples()
        if row.edge_id == "mp_vent"
    }
    records = []
    for row in pack.daily.itertuples():
        observed = pd.Series({e: getattr(row, e) for e in METERED_EDGES} | {"day": row.day})
        vent = vent_lookup.get(row.day)
        db = balance_day(net, observed, pack.meter_daily_bounds,
                         waste_cap_t=float(row.waste_t), vent_supplement=vent)
        rec = {"day": row.day, "status": db.status, "note": db.note,
               "retrofit": row.retrofit, "maintenance": row.maintenance,
               "load": row.load, "service_t": row.service_t}
        for e in EDGE_ORDER:
            rec[f"obs_{e}"] = db.observed[e]
            rec[f"est_{e}"] = db.estimate[e]
            rec[f"lo_{e}"] = db.lower[e]
            rec[f"hi_{e}"] = db.upper[e]
        for e in METERED_EDGES:
            rec[f"dev_{e}"] = db.deviations[e]
        for node, s in db.slack.items():
            rec[f"slack_{node}"] = s
        records.append(rec)
    return pd.DataFrame(records)


def build_network_from_pack(pack) -> Network:
    from .network import build_network
    return build_network(pack.edges)
