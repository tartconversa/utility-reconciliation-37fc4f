"""蒸汽压力网络与公用工程边界定义。

网络（8 小时日窗流量，单位吨；conversion 均为 1）：

    boiler ──boiler_hp──▶ HP ──hp_mp(背压)──▶ MP ──mp_lp(减压)──▶ LP ──lp_user──▶ 生产
                          │                   ▲                    │
                          └──hp_device──▶ 压缩机 │ waste_mp(余热)     └──lp_loss──▶ 损失
                                              MP ──mp_vent──▶ 放空

节点平衡（等式约束）：
    HP: boiler_hp = hp_mp + hp_device
    MP: hp_mp + waste_mp = mp_lp + mp_vent
    LP: mp_lp = lp_user + lp_loss

口径约束（docs/资料口径.md）：
- 设备消耗不能凭空变成 LP 供汽；背压发电按通过 hp_mp 的汽量计。
- 供汽缺口不可用购电替代；外购电只补发电不足，价格与排放线性计量。
- 未计量边只有容量范围，无已知流量，不允许隐含平衡插件。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# 全部边的固定顺序，作为向量变量列序。
EDGE_ORDER = [
    "boiler_hp",
    "hp_mp",
    "mp_lp",
    "waste_mp",
    "hp_device",
    "lp_user",
    "mp_vent",
    "lp_loss",
]

METERED_EDGES = ["boiler_hp", "hp_mp", "mp_lp", "hp_device", "lp_user"]
UNMETERED_EDGES = ["waste_mp", "mp_vent", "lp_loss"]

# 节点平衡：节点 -> {边: 系数}，入为正、出为负，等式取 0。
NODE_BALANCES: dict[str, dict[str, float]] = {
    "HP": {"boiler_hp": +1, "hp_mp": -1, "hp_device": -1},
    "MP": {"hp_mp": +1, "waste_mp": +1, "mp_lp": -1, "mp_vent": -1},
    "LP": {"mp_lp": +1, "lp_user": -1, "lp_loss": -1},
}

# 背压发电所在边。
BACKPRESSURE_EDGE = "hp_mp"


@dataclass(frozen=True)
class Network:
    """边容量与节点平衡的不可变描述。"""

    edges: pd.DataFrame
    edge_order: tuple[str, ...]
    node_balances: dict[str, dict[str, float]]

    @property
    def index(self) -> dict[str, int]:
        return {edge: i for i, edge in enumerate(self.edge_order)}

    def bounds(self, edge_id: str) -> tuple[float, float]:
        row = self.edges.loc[self.edges.edge_id == edge_id].iloc[0]
        return float(row.lower), float(row.upper)

    def capacity_frame(self) -> pd.DataFrame:
        rows = []
        for edge_id in self.edge_order:
            lo, hi = self.bounds(edge_id)
            rows.append({"edge_id": edge_id, "lower": lo, "upper": hi})
        return pd.DataFrame(rows)


def build_network(edges: pd.DataFrame) -> Network:
    """从 edges.csv 构造网络，并校验资料口径中的连接齐备。"""
    known = set(edges.edge_id)
    missing = [e for e in EDGE_ORDER if e not in known]
    if missing:
        raise ValueError(f"edges.csv 缺少口径要求的连接：{missing}")
    if (edges.conversion.astype(float) != 1.0).any():
        raise ValueError("当前模型仅支持 conversion=1 的连接，收到非 1 转换比")
    return Network(
        edges=edges.copy(),
        edge_order=tuple(EDGE_ORDER),
        node_balances={k: dict(v) for k, v in NODE_BALANCES.items()},
    )


def backpressure_generation(hp_mp_t: float, kwh_per_t: float) -> float:
    """背压发电量（千瓦时）= 通过背压机汽量 × 系数。"""
    return float(hp_mp_t) * float(kwh_per_t)
