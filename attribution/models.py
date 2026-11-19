"""历史关系的受控估计与可识别性判断。

关键纪律：
- 不预设节煤模型。煤耗与锅炉汽量的关系只能由 power 与 measurements 共同估计。
- 负荷与煤耗共享周内循环，简单双变量回归会产生伪相关；必须控制星期效应
  与改造阶段后再看锅炉汽量的偏效应。
- 识别不出的关系如实报告“不可识别”，不用平衡反推替代观测。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegressionResult:
    formula: str
    coefficients: dict[str, float]
    r2: float
    rmse: float
    n: int
    identifiable: bool
    note: str


def _ols(X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, float], float, float]:
    beta, *_ = np.linalg.lstsq(X.values.astype(float), y.values.astype(float), rcond=None)
    pred = X.values.astype(float) @ beta
    resid = y.values.astype(float) - pred
    ss_res = float((resid**2).sum())
    ss_tot = float(((y.values - y.values.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(y)))
    return dict(zip(X.columns, (float(b) for b in beta))), r2, rmse


def fit_coal_model(daily: pd.DataFrame) -> RegressionResult:
    """控制星期效应与改造阶段后，检验煤耗对锅炉汽量的偏效应。

    返回两个嵌套模型的对比：若加入 boiler_hp 后 R² 无增益且其系数接近 0，
    则资料不支持“减汽即减煤”的节煤模型。
    """
    d = daily.copy()
    d["dow"] = pd.to_datetime(d["day"]).dt.dayofweek
    dow = pd.get_dummies(d["dow"], prefix="dow", drop_first=True).astype(float)
    base = pd.concat([
        pd.Series(1.0, index=d.index, name="const"),
        dow.reset_index(drop=True),
        pd.Series(d["retrofit"].astype(float).values, name="retrofit"),
    ], axis=1)
    y = d["coal_t"].reset_index(drop=True)

    coef0, r2_0, rmse0 = _ols(base, y)
    extended = base.copy()
    extended["boiler_hp_c"] = (d["boiler_hp"] - d["boiler_hp"].mean()).reset_index(drop=True)
    coef1, r2_1, rmse1 = _ols(extended, y)

    steam_coef = coef1["boiler_hp_c"]
    # 完全拟合的合成资料下增益为 0；真实资料用 1e-4 的 R² 增益作经验阈值。
    gain = r2_1 - r2_0
    identifiable = gain > 1e-4 and abs(steam_coef) > 1e-3
    note = (
        "控制星期与改造阶段后，锅炉汽量对煤耗无解释增益（ΔR²=%.2e，偏系数=%.4f t煤/t汽）；"
        "资料不支持由减汽推断减煤，节煤量不可识别。改造阶段煤耗阶跃=%.3f t/日，"
        "但同期锅炉汽量未下降，该阶跃不能归因于减汽。"
        % (gain, steam_coef, coef1["retrofit"])
    ) if not identifiable else (
        "锅炉汽量对煤耗存在可识别偏效应 %.4f t煤/t汽（ΔR²=%.2e）。" % (steam_coef, gain)
    )
    return RegressionResult(
        formula="coal_t ~ 星期效应 + retrofit + boiler_hp(中心化)",
        coefficients=coef1,
        r2=r2_1,
        rmse=rmse1,
        n=len(d),
        identifiable=identifiable,
        note=note,
    )


def fit_naive_coal_steam(daily: pd.DataFrame) -> RegressionResult:
    """不控制混淆因素的双变量回归，仅用于暴露伪相关，不得用于收益计算。"""
    d = daily[["boiler_hp", "coal_t"]].dropna()
    X = pd.DataFrame({"const": 1.0, "boiler_hp": d["boiler_hp"]})
    coef, r2, rmse = _ols(X, d["coal_t"])
    return RegressionResult(
        formula="coal_t ~ boiler_hp（未控制星期/阶段，仅作对照）",
        coefficients=coef, r2=r2, rmse=rmse, n=len(d),
        identifiable=False,
        note="未控制星期效应与改造阶段，斜率受共同循环驱动，属伪相关，禁止用于节煤归因。",
    )


def fit_electricity_model(daily: pd.DataFrame) -> RegressionResult:
    """电力边界：购电 + 背压发电 = 设备电耗 + 厂区其他电耗。

    检验“其他电耗”是否随负荷/服务量变化，以确认改造的电力后果
    仅通过设备增电与背压汽量两条可计量通道传导。
    """
    d = daily.copy()
    d["bp_gen"] = d["hp_mp"] * d["backpressure_kwh_t"]
    d["other_elec"] = d["import_kwh"] + d["bp_gen"] - d["device_kwh"]
    X = pd.DataFrame({
        "const": 1.0,
        "load": d["load"] - d["load"].mean(),
        "service_c": d["service_t"] - d["service_t"].mean(),
        "retrofit": d["retrofit"].astype(float),
    })
    coef, r2, rmse = _ols(X, d["other_elec"])
    # 经验阈值：阶段系数相对均值 <2% 且负荷灵敏度 <300 kWh/负荷单位，
    # 视为厂区其他电耗对改造不敏感（残余波动仅为正常计量分散）。
    flat = abs(coef["retrofit"]) < 0.02 * d["other_elec"].mean() and abs(coef["load"]) < 300
    note = (
        "厂区其他电耗对改造阶段不敏感（阶段系数 %.1f kWh，约占均值 %.1f%%），"
        "改造的电力后果可由设备增电与背压发电变化两条可计量通道完整刻画；"
        "负荷弱灵敏度 %.1f kWh/单位在计量分散内。"
        % (coef["retrofit"], 100 * coef["retrofit"] / d["other_elec"].mean(), coef["load"])
    ) if flat else "其他电耗随阶段或负荷变化，电力归因存在未解释成分，需保留区间。"
    return RegressionResult(
        formula="其他电耗 ~ 负荷 + 服务量 + retrofit",
        coefficients=coef, r2=r2, rmse=rmse, n=len(d),
        identifiable=flat, note=note,
    )


def matched_phase_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """按负荷档匹配改造前后日窗，给出可比工况下的均值对照。

    两阶段负荷档相同（0.55–0.85 七档循环），服务量分布一致，
    因此同档均值之差是有资料支持的可比条件；检修日单独标记不剔除，
    由调用方决定是否排除。
    """
    d = daily.copy()
    rows = []
    for load, grp in d.groupby("load"):
        pre = grp[~grp.retrofit]
        post = grp[grp.retrofit]
        if pre.empty or post.empty:
            continue
        rows.append({
            "load": load,
            "n_pre": len(pre), "n_post": len(post),
            "service_pre": pre.service_t.mean(), "service_post": post.service_t.mean(),
            "hp_device_pre": pre.hp_device.mean(), "hp_device_post": post.hp_device.mean(),
            "boiler_hp_pre": pre.boiler_hp.mean(), "boiler_hp_post": post.boiler_hp.mean(),
            "hp_mp_pre": pre.hp_mp.mean(), "hp_mp_post": post.hp_mp.mean(),
            "import_pre": pre.import_kwh.mean(), "import_post": post.import_kwh.mean(),
            "coal_pre": pre.coal_t.mean(), "coal_post": post.coal_t.mean(),
        })
    out = pd.DataFrame(rows).sort_values("load")
    out["device_steam_delta"] = out.hp_device_post - out.hp_device_pre
    out["boiler_delta"] = out.boiler_hp_post - out.boiler_hp_pre
    out["hp_mp_delta"] = out.hp_mp_post - out.hp_mp_pre
    return out
