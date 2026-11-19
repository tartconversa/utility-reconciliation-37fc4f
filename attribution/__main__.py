"""命令行入口：资料检查、平衡诊断、方案评审、承诺检验与情景对比。

用法：
    python -m attribution checksum --root .
    python -m attribution balance --root . [--as-of 2025-02-05T08:00:00+08:00]
    python -m attribution review  --root . [--scenario central] [--as-of ...]
    python -m attribution verify  --root . [--proposal A]
    python -m attribution scenarios --root .
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 支持 `python -I attribution/__main__.py` 直接执行（隔离模式下 cwd 不在 sys.path）。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "attribution"

import pandas as pd

from . import balance as balance_mod
from . import dataio, eval as eval_mod, report as report_mod, sensitivity, verify as verify_mod


def _root(args) -> Path:
    return Path(args.root).resolve()


def cmd_checksum(args) -> None:
    """资料摘要核对：catalog 登记值与实际文件不一致时明确报告，不静默通过。"""
    import hashlib

    root = _root(args)
    catalog = json.loads((root / "data" / "catalog.json").read_text(encoding="utf-8"))
    ok = True
    for table in catalog["tables"]:
        path = root / table["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        match = actual == table["sha256"]
        ok &= match
        print(f"{'OK  ' if match else 'DIFF'} {table['name']:14s} "
              f"catalog={table['sha256'][:16]}… actual={actual[:16]}…")
    print("RESULT:", "consistent" if ok else "MISMATCH（catalog 摘要与 shipped CSV 不符，分析以原始 CSV 为准）")


def _load(args):
    return dataio.load_pack(_root(args), as_of=getattr(args, "as_of", None))


def cmd_balance(args) -> None:
    pack = _load(args)
    df = balance_mod.balance_pack(pack)
    diag = report_mod.balance_diagnosis_table(df)
    out = Path(args.out) if args.out else None
    if out:
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"写入 {out}（{len(df)} 行）")
    print(diag.to_string(index=False))
    print("\n状态计数：", df.status.value_counts().to_dict())


def cmd_review(args) -> None:
    pack = _load(args)
    result, diag = eval_mod.run_review(pack, scenario=args.scenario)
    balance_df = balance_mod.balance_pack(pack)
    checks = None
    if pack.daily.retrofit.nunique() > 1:
        checks = verify_mod.verify_commitments(pack, "A")
    text = report_mod.text_report(pack, args.scenario, result, diag, balance_df, checks)
    print(text)
    if args.out:
        out = Path(args.out)
        result.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n结果表写入 {out}")


def cmd_verify(args) -> None:
    pack = _load(args)
    checks = verify_mod.verify_commitments(pack, args.proposal)
    for c in checks:
        print(f"- {c.item}：承诺={c.committed} 观测=[{c.observed_lo:.2f}, {c.observed_hi:.2f}]"
              f" {c.unit} => {c.verdict}")
        print(f"    {c.detail}")
    if args.timeline:
        tl = verify_mod.review_as_of(_root(args), args.proposal)
        print("\n证据时点时间线：")
        print(tl.assign(as_of=tl.as_of.astype(str)).to_string(index=False))


def cmd_scenarios(args) -> None:
    pack = _load(args)
    df = report_mod.scenario_sweep(pack)
    cols = ["scenario", "proposal_id", "coal_status",
            "delta_import_kwh_lo", "delta_import_kwh_hi",
            "delta_cost_yuan_lo", "delta_cost_yuan_hi",
            "delta_emission_kg_lo", "delta_emission_kg_hi"]
    print(df[cols].round(2).to_string(index=False))
    if args.out:
        df.to_csv(Path(args.out), index=False, encoding="utf-8-sig")
        print(f"\n写入 {args.out}")


def cmd_rank(args) -> None:
    pack = _load(args)
    result, _ = eval_mod.run_review(pack, args.scenario)
    print(sensitivity.rank_proposals(result, args.metric).to_string(index=False))


def cmd_decision(args) -> None:
    pack = _load(args)
    print(sensitivity.decision_summary(pack, args.scenario).to_string(index=False))


def cmd_tighten(args) -> None:
    """仪表误差收紧后的重评：对比可行平衡、不可平衡日与排序的变化。"""
    pack = _load(args)
    override = {}
    for item in args.bound or []:
        edge_id, bound = item.split("=", 1)
        override[edge_id.strip()] = float(bound)
    if not override:
        raise SystemExit("请用 --bound edge_id=单读数误差界 至少指定一项，如 --bound hp_device=0.75")
    tightened = sensitivity.with_meter_bounds(pack, override)

    r0, d0 = eval_mod.run_review(pack, args.scenario)
    r1, d1 = eval_mod.run_review(tightened, args.scenario)
    print("误差界收紧（单读数界）：", override)
    print("\n[不可平衡日数变化]")
    cmp_df = r0[["proposal_id", "conflict_days", "mean_slack_t"]].merge(
        r1[["proposal_id", "conflict_days", "mean_slack_t"]],
        on="proposal_id", suffixes=("_原", "_收紧"))
    print(cmp_df.round(3).to_string(index=False))
    print("\n[费用区间变化，元/日]")
    cost = r0[["proposal_id", "delta_cost_yuan_lo", "delta_cost_yuan_hi"]].merge(
        r1[["proposal_id", "delta_cost_yuan_lo", "delta_cost_yuan_hi"]],
        on="proposal_id", suffixes=("_原", "_收紧"))
    print(cost.round(1).to_string(index=False))
    print("\n[排序]")
    print(sensitivity.rank_proposals(r1, "delta_cost_yuan").to_string(index=False))
    print("\n变化来源：测量误差界收紧（物理推断），外部价格未变。")


def cmd_decompose(args) -> None:
    pack = _load(args)
    print(sensitivity.measurement_vs_price_decomposition(pack).round(2).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="attribution", description="汽改电公用工程联合归因")
    p.add_argument("--root", default=".", help="资料包根目录")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("checksum", help="核对 catalog 摘要")
    sp.set_defaults(func=cmd_checksum)

    sp = sub.add_parser("balance", help="逐日可行平衡与冲突诊断")
    sp.add_argument("--as-of", default=None)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_balance)

    sp = sub.add_parser("review", help="方案评审（单项+组合，联合归因）")
    sp.add_argument("--scenario", default="central",
                    choices=["central", "low_grid", "peak_price"])
    sp.add_argument("--as-of", default=None)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_review)

    sp = sub.add_parser("verify", help="改造后承诺检验")
    sp.add_argument("--proposal", default="A")
    sp.add_argument("--timeline", action="store_true")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("scenarios", help="外部价格/排放情景对比")
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_scenarios)

    sp = sub.add_parser("rank", help="方案排序及区间重叠稳健性")
    sp.add_argument("--scenario", default="central",
                    choices=["central", "low_grid", "peak_price"])
    sp.add_argument("--metric", default="delta_cost_yuan",
                    choices=["delta_cost_yuan", "delta_emission_kg", "delta_import_kwh"])
    sp.set_defaults(func=cmd_rank)

    sp = sub.add_parser("decision", help="投资决策证据状态视图")
    sp.add_argument("--scenario", default="central",
                    choices=["central", "low_grid", "peak_price"])
    sp.set_defaults(func=cmd_decision)

    sp = sub.add_parser("tighten", help="仪表误差收紧后的重评对比")
    sp.add_argument("--scenario", default="central",
                    choices=["central", "low_grid", "peak_price"])
    sp.add_argument("--as-of", default=None)
    sp.add_argument("--bound", action="append",
                    help="edge_id=单读数误差界，可重复，如 --bound hp_device=0.75 --bound hp_mp=1.0")
    sp.set_defaults(func=cmd_tighten)

    sp = sub.add_parser("decompose", help="测量变化与外部价格变化的来源分解")
    sp.set_defaults(func=cmd_decompose)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
