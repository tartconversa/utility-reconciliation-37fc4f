"""资料只读载入与证据可知时间过滤。

口径来源：docs/资料口径.md。
- 空单元表示未知或不适用，不等于零。
- 读数是半开区间总量；4 小时分表按实际区间对齐到 8 小时日窗，不做乘除。
- bound 是对该条观测本身生效的绝对误差界，不是标准差；同日多条分表误差界相加。
- as_of 过滤只影响“评审当时可知”的资料，不删除或修改任何原始观测。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import datetime as _dt

import pandas as pd

# 日窗定义：measurements 中每日观测窗为北京时间 08:00 起的 8 小时。
WINDOW_HOURS = 8
WINDOW_START_HOUR = 8

# 有计量仪表的边（meter_id -> edge_id），顺序同时用于列名约定。
METER_EDGES = {
    "M1": "boiler_hp",
    "M2": "hp_mp",
    "M3": "mp_lp",
    "M4": "hp_device",
    "M5": "lp_user",
}
EDGE_METERS = {v: k for k, v in METER_EDGES.items()}

# 未计量边，容量来自 edges.csv，实际流量未知。
UNMETERED_EDGES = ["mp_vent", "lp_loss"]
RECOVERY_EDGE = "waste_mp"


def _parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert("Asia/Shanghai")


def _as_of_timestamp(as_of: str | _dt.datetime | pd.Timestamp | None) -> pd.Timestamp | None:
    if as_of is None:
        return None
    ts = pd.to_datetime(as_of)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Shanghai")
    else:
        ts = ts.tz_convert("Asia/Shanghai")
    return ts


@dataclass(frozen=True)
class DataPack:
    """全部资料的只读容器，按一个 as_of 时点切片。"""

    root: Path
    as_of: pd.Timestamp | None
    edges: pd.DataFrame
    meter_specs: pd.DataFrame
    measurements: pd.DataFrame          # 原始读数，未聚合、未修改
    operations: pd.DataFrame
    power: pd.DataFrame
    proposals: pd.DataFrame
    external: pd.DataFrame
    supplement: pd.DataFrame
    daily: pd.DataFrame                 # 对齐到 8 小时日窗的聚合表
    meter_daily_bounds: dict[str, float]
    excluded_by_as_of: dict[str, int]

    def edge_capacity(self, edge_id: str) -> tuple[float, float]:
        row = self.edges.loc[self.edges.edge_id == edge_id].iloc[0]
        return float(row.lower), float(row.upper)


def _filter_as_of(df: pd.DataFrame, as_of: pd.Timestamp | None, col: str) -> tuple[pd.DataFrame, int]:
    if as_of is None or col not in df.columns:
        return df, 0
    mask = df[col] <= as_of
    return df.loc[mask].copy(), int((~mask).sum())


def load_pack(root: str | Path, as_of: str | pd.Timestamp | None = None) -> DataPack:
    """载入资料包并按证据可知时间切片。

    as_of=None 表示使用全部资料（事后复核口径）；给定时间戳时，
    available_at 晚于该时点的读数与补测不进入分析，但计数保留在
    excluded_by_as_of 中，保证评审依据可重现。
    """
    root = Path(root)
    as_of_ts = _as_of_timestamp(as_of)

    edges = pd.read_csv(root / "data" / "edges.csv")
    meter_specs = pd.read_csv(root / "data" / "meter_specs.csv")
    measurements = pd.read_csv(root / "data" / "measurements.csv")
    operations = pd.read_csv(root / "data" / "operations.csv")
    power = pd.read_csv(root / "data" / "power.csv")
    proposals = pd.read_csv(root / "data" / "proposals.csv")
    external = pd.read_csv(root / "data" / "external.csv")
    supplement = pd.read_csv(root / "data" / "supplement.csv")

    measurements["start"] = _parse_ts(measurements["start"])
    measurements["available_at"] = _parse_ts(measurements["available_at"])
    operations["day"] = pd.to_datetime(operations["day"])
    operations["available_at"] = _parse_ts(operations["available_at"])
    power["day"] = pd.to_datetime(power["day"])
    supplement["day"] = pd.to_datetime(supplement["day"])
    supplement["available_at"] = _parse_ts(supplement["available_at"])

    excluded: dict[str, int] = {}
    measurements, n1 = _filter_as_of(measurements, as_of_ts, "available_at")
    supplement, n2 = _filter_as_of(supplement, as_of_ts, "available_at")
    operations, n3 = _filter_as_of(operations, as_of_ts, "available_at")
    excluded.update(measurements=n1, supplement=n2, operations=n3)

    daily, meter_daily_bounds = _build_daily(measurements, meter_specs)
    daily = (
        daily.merge(operations, on="day", how="inner")
        .merge(power, on="day", how="inner")
    )
    daily = daily.sort_values("day").reset_index(drop=True)

    return DataPack(
        root=root.resolve(),
        as_of=as_of_ts,
        edges=edges,
        meter_specs=meter_specs,
        measurements=measurements,
        operations=operations,
        power=power,
        proposals=proposals,
        external=external,
        supplement=supplement,
        daily=daily,
        meter_daily_bounds=meter_daily_bounds,
        excluded_by_as_of=excluded,
    )


def _build_daily(
    measurements: pd.DataFrame, meter_specs: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """把不同采样间隔的读数对齐到 8 小时日窗并求和。

    4 小时 value 是该 4 小时的实际总量，直接与同窗另一条相加，
    不做任何乘除（资料口径明确禁止再次缩放凑平衡）。
    """
    spec = meter_specs.set_index("meter_id")
    frames = []
    daily_bounds: dict[str, float] = {}
    for meter_id, edge_id in METER_EDGES.items():
        sub = measurements[measurements.meter_id == meter_id].copy()
        if sub.empty:
            continue
        # 日窗键：08:00 读数归入其自然日；校验同日读数覆盖的总时长一致。
        sub["day"] = sub["start"].dt.tz_convert(None).dt.normalize()
        hours_per_day = sub.groupby("day").hours.sum()
        interval_h = int(spec.loc[meter_id, "interval_h"])
        expected_count = WINDOW_HOURS // interval_h
        if not (hours_per_day == WINDOW_HOURS).all():
            bad = hours_per_day[hours_per_day != WINDOW_HOURS]
            raise ValueError(
                f"仪表 {meter_id} 存在未覆盖完整 {WINDOW_HOURS} 小时日窗的记录：{bad.to_dict()}"
            )
        counts = sub.groupby("day").size()
        if not (counts == expected_count).all():
            raise ValueError(f"仪表 {meter_id} 每日读数条数与 {interval_h} 小时采样约定不符")
        agg = sub.groupby("day", as_index=False).value.sum()
        agg = agg.rename(columns={"value": edge_id})
        frames.append(agg)
        # 同日 n 条读数的绝对误差界相加（界对每条观测本身生效）。
        daily_bounds[edge_id] = float(spec.loc[meter_id, "bound"]) * expected_count

    daily = frames[0]
    for frame in frames[1:]:
        daily = daily.merge(frame, on="day", how="outer")
    return daily.sort_values("day").reset_index(drop=True), daily_bounds
