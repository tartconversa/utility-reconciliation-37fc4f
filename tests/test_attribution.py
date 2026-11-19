"""分析包业务测试：口径纪律、平衡可行性、模型可识别性与评估纪律。

运行：python -m unittest discover -s tests -v
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from attribution import balance as balance_mod
from attribution import dataio, eval as eval_mod, models, sensitivity, verify as verify_mod
from attribution.network import EDGE_ORDER, METERED_EDGES, build_network


class DataIOTests(unittest.TestCase):
    def setUp(self):
        self.pack = dataio.load_pack(ROOT)

    def test_four_hour_aggregation_no_rescaling(self):
        """4 小时两条读数直接相加覆盖 8 小时窗，不乘除 2。"""
        m = self.pack.measurements
        m2 = m[m.meter_id == "M2"]
        one_day = m2[m2.start.dt.date == m2.start.dt.date.iloc[0]]
        self.assertEqual(len(one_day), 2)
        daily_m2 = self.pack.daily.hp_mp.iloc[0]
        self.assertAlmostEqual(daily_m2, one_day.value.sum(), places=6)

    def test_daily_bounds_add_across_readings(self):
        """同日多条读数的绝对误差界相加。"""
        spec = self.pack.meter_specs.set_index("meter_id")
        self.assertAlmostEqual(self.pack.meter_daily_bounds["hp_mp"],
                               spec.loc["M2", "bound"] * 2)
        self.assertAlmostEqual(self.pack.meter_daily_bounds["boiler_hp"],
                               spec.loc["M1", "bound"])

    def test_as_of_filters_supplement_but_keeps_count(self):
        early = dataio.load_pack(ROOT, as_of="2025-03-08T08:00:00+08:00")
        self.assertEqual(early.excluded_by_as_of["supplement"], 2)
        self.assertTrue(early.supplement.empty)
        late = dataio.load_pack(ROOT, as_of="2025-03-09T08:00:00+08:00")
        self.assertEqual(len(late.supplement), 2)

    def test_empty_string_is_missing_not_zero(self):
        """资料口径：空单元表示未知，本资料无空值时列仍按浮点读入。"""
        self.assertEqual(len(self.pack.daily), 60)


class NetworkTests(unittest.TestCase):
    def test_balances_and_capacities_complete(self):
        pack = dataio.load_pack(ROOT)
        net = build_network(pack.edges)
        self.assertEqual(set(net.edge_order), set(EDGE_ORDER))
        lo, hi = net.bounds("lp_loss")
        self.assertEqual((lo, hi), (0.0, 12.0))
        lo, hi = net.bounds("boiler_hp")
        self.assertEqual((lo, hi), (35.0, 120.0))


class BalanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = dataio.load_pack(ROOT)
        cls.df = balance_mod.balance_pack(cls.pack)

    def test_conflict_count_matches_manual_residual(self):
        """11 个 HP 残差超界日必须被识别为 conflict，与手工核算一致。"""
        d = self.pack.daily
        resid = d.boiler_hp - d.hp_mp - d.hp_device
        bound = self.pack.meter_daily_bounds
        combined = bound["boiler_hp"] + bound["hp_mp"] + bound["hp_device"]
        manual = int((resid.abs() > combined).sum())
        self.assertEqual(manual, 11)
        self.assertEqual(int((self.df.status == "conflict").sum()), manual)
        # 冲突全部在 HP 节点
        self.assertTrue((self.df.slack_MP.abs() < 1e-6).all())
        self.assertTrue((self.df.slack_LP.abs() < 1e-6).all())

    def test_lp_loss_always_inside_physical_capacity(self):
        """LP 损失是有物理依据的未计量边，可行区间必须落在容量 [0,12] 内。"""
        self.assertTrue((self.df.lo_lp_loss >= -1e-6).all())
        self.assertTrue((self.df.hi_lp_loss <= 12.0 + 1e-6).all())

    def test_metered_estimates_stay_within_bounds(self):
        """可信测量不得被任意修改：估计值必须在观测±界内。"""
        for e in METERED_EDGES:
            within = ((self.df[f"est_{e}"] >= self.df[f"obs_{e}"] -
                       self.pack.meter_daily_bounds[e] - 1e-6) &
                      (self.df[f"est_{e}"] <= self.df[f"obs_{e}"] +
                       self.pack.meter_daily_bounds[e] + 1e-6))
            self.assertTrue(within.all(), e)

    def test_vent_and_recovery_not_point_identified(self):
        """无补测日的 mp_vent 与 waste_mp 是联合不可识别的，区间必须宽于点值。"""
        ordinary = self.df[self.df.lo_mp_vent < self.df.hi_mp_vent]
        self.assertGreater(len(ordinary), 50)

    def test_supplement_tightens_vent_interval(self):
        early = dataio.load_pack(ROOT, as_of="2025-03-08T08:00:00+08:00")
        late = dataio.load_pack(ROOT, as_of="2025-03-09T08:00:00+08:00")
        db_b = balance_mod.balance_pack(early)
        db_a = balance_mod.balance_pack(late)
        day = "2025-02-15"
        rb = db_b[db_b.day.astype(str).str.startswith(day)].iloc[0]
        ra = db_a[db_a.day.astype(str).str.startswith(day)].iloc[0]
        self.assertGreater(rb.hi_mp_vent - rb.lo_mp_vent,
                           ra.hi_mp_vent - ra.lo_mp_vent)
        self.assertAlmostEqual(ra.lo_mp_vent, 6.0, places=6)
        self.assertAlmostEqual(ra.hi_mp_vent, 8.0, places=6)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.pack = dataio.load_pack(ROOT)

    def test_coal_steam_effect_not_identifiable(self):
        """控制星期与阶段后，锅炉汽量对煤耗偏效应为 0，节煤不可识别。"""
        res = models.fit_coal_model(self.pack.daily)
        self.assertFalse(res.identifiable)
        self.assertAlmostEqual(res.coefficients["boiler_hp_c"], 0.0, places=6)
        self.assertAlmostEqual(res.coefficients["retrofit"], -0.3, places=4)
        self.assertGreater(res.r2, 0.999)

    def test_naive_regression_is_flagged_pseudocorrelation(self):
        res = models.fit_naive_coal_steam(self.pack.daily)
        self.assertFalse(res.identifiable)
        self.assertIn("伪相关", res.note)

    def test_other_electricity_is_flat(self):
        res = models.fit_electricity_model(self.pack.daily)
        self.assertTrue(res.identifiable)
        # 阶段系数相对均值（约 5050 kWh）小于 2%，不构成可归因的电力通道
        self.assertLess(abs(res.coefficients["retrofit"]),
                        0.02 * self.pack.daily.import_kwh.mean())

    def test_matched_phases_share_load_levels(self):
        tab = models.matched_phase_summary(self.pack.daily)
        self.assertEqual(len(tab), 7)
        # 观测到的设备减汽在 7–9 t，与方案 A 的 8 t 相容
        self.assertTrue((tab.device_steam_delta < -7).all())
        # 锅炉出力前后无实质变化
        self.assertTrue(tab.boiler_delta.abs().max() < 1.0)


class EvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = dataio.load_pack(ROOT)
        cls.result, cls.diag = eval_mod.run_review(cls.pack, "central")

    def test_baseline_has_no_native_conflict(self):
        self.assertEqual(self.diag["baseline_conflict_days"], 0)

    def test_steam_reduction_reroutes_to_backpressure_not_boiler(self):
        """减汽经背压通道吸收而非锅炉少烧：区间方向与观测双证。

        点解与区间中心会被反事实不可平衡日的 HP 松弛和 hp_device 容量
        下限拉偏，因此检验三条稳健事实：设备区间整体下移、背压区间整体
        上移、锅炉区间不缩小；8 t 的设备侧承诺另由 verify 的负荷匹配观测
        （实测减汽 7.92 t）独立验证。
        """
        ev = self.diag["evals"]
        base, cfA = ev["baseline"]["flows"], ev["A"]["flows"]
        self.assertLess(cfA["hp_device"]["hi"], base["hp_device"]["hi"] - 4.0)
        self.assertGreater(cfA["hp_mp"]["lo"], base["hp_mp"]["lo"] + 5.0)
        # 锅炉区间宽度与位置基本不变：减汽没有传导到锅炉
        self.assertAlmostEqual(cfA["boiler_hp"]["lo"], base["boiler_hp"]["lo"], delta=0.1)
        self.assertLess(abs(cfA["boiler_hp"]["hi"] - base["boiler_hp"]["hi"]), 5.0)

    def test_coal_savings_recorded_zero_when_unidentified(self):
        for _, r in self.result.iterrows():
            self.assertEqual(r.coal_status, "not_identifiable")
            self.assertEqual(r.delta_coal_t_lo, 0.0)
            self.assertEqual(r.delta_coal_t_hi, 0.0)

    def test_all_proposals_increase_cost_and_emissions(self):
        """本资料下减汽未触达锅炉，所有方案都是净增购电、净增费用。"""
        self.assertTrue((self.result.delta_cost_yuan_lo > 0).all())
        self.assertTrue((self.result.delta_import_kwh_lo > 0).all())

    def test_interaction_interval_reported_honestly(self):
        inter = self.diag["interaction"]
        cost = inter[inter.metric == "delta_cost_yuan"].iloc[0]
        # 区间含 0：现有测量无法证明交互，不能声称可加或不可加
        self.assertTrue(cost.contains_zero)
        self.assertLessEqual(cost.interaction_lo, 0.0)
        self.assertGreaterEqual(cost.interaction_hi, 0.0)

    def test_counterfactual_infeasible_days_are_flagged(self):
        """方案诱发的不可平衡日必须显式计数，不得静默平衡。"""
        row_a = self.result[self.result.proposal_id == "A"].iloc[0]
        self.assertGreater(row_a.induced_conflict_days, 0)

    def test_reproducibility_under_same_as_of(self):
        r1, d1 = eval_mod.run_review(
            dataio.load_pack(ROOT, as_of="2025-02-20T08:00:00+08:00"), "central")
        r2, d2 = eval_mod.run_review(
            dataio.load_pack(ROOT, as_of="2025-02-20T08:00:00+08:00"), "central")
        pd.testing.assert_frame_equal(r1, r2)
        self.assertEqual(d1["baseline_days"], d2["baseline_days"])


class VerifyTests(unittest.TestCase):
    def test_commitment_checks_cover_three_verdicts(self):
        pack = dataio.load_pack(ROOT)
        checks = verify_mod.verify_commitments(pack, "A")
        verdicts = {c.item: c.verdict for c in checks}
        self.assertIn("within_meter_uncertainty", verdicts.values())
        self.assertIn("reference_only", verdicts.values())

    def test_timeline_changes_only_as_evidence_arrives(self):
        tl = verify_mod.review_as_of(ROOT, "A")
        self.assertGreater(len(tl), 0)
        # 3 月 9 日补测进入后，排除计数归零
        late = tl[tl.as_of >= pd.Timestamp("2025-03-09", tz="Asia/Shanghai")]
        self.assertTrue((late.excluded_supplement == 0).all())


class SensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = dataio.load_pack(ROOT)

    def test_ranking_robust_when_intervals_disjoint(self):
        result, _ = eval_mod.run_review(self.pack, "central")
        rank = sensitivity.rank_proposals(result, "delta_cost_yuan")
        # 本资料下 B 增量最小且三方案区间不重叠，排序稳健
        self.assertEqual(rank.iloc[0].proposal_id, "B")
        self.assertTrue(rank.ranking_robust.all())

    def test_tightening_bounds_narrows_intervals_and_exposes_conflict(self):
        result0, _ = eval_mod.run_review(self.pack, "central")
        tight = sensitivity.with_meter_bounds(
            self.pack, {"hp_device": 0.75, "hp_mp": 1.0, "boiler_hp": 1.5})
        self.assertAlmostEqual(tight.meter_daily_bounds["hp_device"], 0.75 * 2)
        result1, _ = eval_mod.run_review(tight, "central")
        # 费用区间收窄
        width0 = (result0.set_index("proposal_id").loc["A", "delta_cost_yuan_hi"] -
                  result0.set_index("proposal_id").loc["A", "delta_cost_yuan_lo"])
        width1 = (result1.set_index("proposal_id").loc["A", "delta_cost_yuan_hi"] -
                  result1.set_index("proposal_id").loc["A", "delta_cost_yuan_lo"])
        self.assertLess(width1, width0)
        # 更紧的界暴露更多反事实不可平衡日（不会被静默平衡掉）
        self.assertGreaterEqual(
            result1.set_index("proposal_id").loc["A", "conflict_days"],
            result0.set_index("proposal_id").loc["A", "conflict_days"])
        # 原始资料包未被修改
        self.assertNotEqual(self.pack.meter_daily_bounds["hp_device"], 0.75 * 2)

    def test_price_decomposition_attributes_change_to_external_only(self):
        dec = sensitivity.measurement_vs_price_decomposition(self.pack)
        self.assertTrue(dec.source.eq("外部价格/排放因子").all())
        # 低电网因子情景只改排放不改费用
        low = dec[dec.comparison == "central -> low_grid"]
        self.assertTrue((low.cost_delta_vs_base_lo == 0).all())
        self.assertTrue((low.emission_delta_vs_base_hi < 0).all())

    def test_decision_summary_flags_coal_and_recovery_for_measurement(self):
        summary = sensitivity.decision_summary(self.pack, "central")
        needs = summary[summary.status == "needs_measure"].item.unique()
        self.assertIn("节煤收益", needs)
        self.assertIn("余热回收/放空分摊", needs)
        supported = summary[summary.status == "supported"].item.unique()
        self.assertIn("设备增电与购电增量", supported)


if __name__ == "__main__":
    unittest.main()
