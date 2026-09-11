"""对 C 题原始附件做描述性检查，不修改数据或选择清洗/调度策略。"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json

import numpy as np
import openpyxl

from prepare_data import ROOT, EXPECTED_DATES, EXPECTED_LABELS, clock_label, date_label, read_daily, rows, verify_snapshot


def stats(a):
    a = np.asarray(a, dtype=float)
    return {"n": a.size, "min": float(a.min()), "p25": float(np.quantile(a, .25)),
            "median": float(np.median(a)), "p75": float(np.quantile(a, .75)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max()), "mean": float(a.mean()),
            "zero_n": int((a == 0).sum()), "negative_n": int((a < 0).sum()),
            "nonfinite_n": int((~np.isfinite(a)).sum())}


def position(a, kind):
    flat = np.argmax(a) if kind == "max" else np.argmin(a)
    j, i = np.unravel_index(flat, a.shape)
    return {"date": EXPECTED_DATES[j], "label": EXPECTED_LABELS[i],
            "excel_row": int(j + 2), "excel_col": int(i + 2), "value": float(a[j, i])}


def error_stats(items):
    if not items:
        return {"n": 0}
    err = np.asarray([r["pred"] - r["actual"] for r in items])
    return {"n": len(items), "mae_kw": float(np.abs(err).mean()),
            "rmse_kw": float(np.sqrt(np.mean(err**2))), "bias_kw": float(err.mean())}


def main():
    verify_snapshot()
    manifest = json.loads((ROOT / "data/raw/manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        p = Path(manifest["source_directory"]) / item["file"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"桌面来源已更新，请先核实：{p}")

    baseline_table = rows("附件1.xlsx", "Sheet1")
    baseline = np.asarray([r[1:] for r in baseline_table[1:]], float)
    load = np.asarray(read_daily("附件2.xlsx", "小区负载"), float)
    pv = np.asarray(read_daily("附件2.xlsx", "光伏发电实际功率"), float)
    price = np.asarray(read_daily("附件4.xlsx", "Sheet1"), float)
    dates = [datetime.fromisoformat(d) for d in EXPECTED_DATES]
    months = np.asarray([d.month for d in dates])
    forecast_table = rows("附件3.xlsx", "Sheet1")

    cell_types = {}
    for filename in ("附件1.xlsx", "附件2.xlsx", "附件3.xlsx", "附件4.xlsx"):
        wb = openpyxl.load_workbook(ROOT / "data/raw/附件" / filename, read_only=True, data_only=False)
        try:
            for s in wb:
                counts = Counter(c.data_type for row in s for c in row)
                cell_types[f"{filename}/{s.title}"] = dict(counts)
        finally:
            wb.close()

    # 仅按题目原始时间标签匹配整点，用于诊断；不确定10分钟区间起止，不插值。
    actual_lookup = {(d + timedelta(minutes=(i + 1) * 10)).isoformat(): float(pv[j, i])
                     for j, d in enumerate(dates) for i in range(144)}
    observations, current_date, forecast_keys = [], None, set()
    missing_target, date_fill = [], 0
    for ridx, row in enumerate(forecast_table[1:], 2):
        if row[0] not in (None, ""):
            current_date = date_label(row[0])
        else:
            date_fill += 1
        issued = datetime.fromisoformat(f"{current_date}T{clock_label(row[1])}:00")
        for lead, value in enumerate(row[2:], 1):
            valid = (issued + timedelta(hours=lead)).isoformat()
            key = (issued.isoformat(), lead)
            if key in forecast_keys:
                raise ValueError(f"重复预报：{key}")
            forecast_keys.add(key)
            item = {"issued_at": issued.isoformat(), "lead": lead, "valid_at": valid,
                    "pred": float(value), "excel_row": ridx, "excel_col": lead + 2}
            if valid not in actual_lookup:
                missing_target.append(item)
            else:
                item["actual"] = actual_lookup[valid]
                observations.append(item)

    forecasts = np.asarray([r[2:] for r in forecast_table[1:]], float)
    monthly = [{"month": m, "load_mean_kw": float(load[months == m].mean()),
                "pv_mean_kw": float(pv[months == m].mean()), "price_mean": float(price[months == m].mean()),
                "pv_daily_peak_mean_kw": float(pv[months == m].max(axis=1).mean())} for m in range(1, 13)]
    extrema = {name: {kind: position(a, kind) for kind in ("min", "max")}
               for name, a in (("load", load), ("pv", pv), ("price", price))}
    issue_types = dict(Counter(type(r[0]).__name__ for r in baseline_table[1:]))
    # 同一目标时刻比较较近、较远两个预报；只做描述，不替代购电策略回测。
    groups = {}
    for item in observations:
        groups.setdefault(item["valid_at"], []).append(item)
    pairs = []
    for valid, group in groups.items():
        if group[0]["actual"] <= 0 or len(group) < 2:
            continue
        group.sort(key=lambda r: r["lead"])
        near, far = group[0], group[-1]
        pairs.append((abs(near["pred"] - near["actual"]), abs(far["pred"] - far["actual"])))
    pair_a = np.asarray(pairs)
    nonzero_daylight_mismatch = [r for r in observations if (r["actual"] == 0) != (r["pred"] == 0)]
    summary = {
        "scope": "2025全年描述性检查，不能作为任何训练或预测效果的无泄漏验证",
        "baseline_time_python_types": issue_types, "input_excel_cell_types": cell_types,
        "stats": {"baseline_price": stats(baseline[:, 0]), "baseline_load": stats(baseline[:, 1]),
                  "baseline_pv": stats(baseline[:, 2]), "load": stats(load), "pv_actual": stats(pv),
                  "pv_forecast": stats(forecasts), "price": stats(price)},
        "extrema": extrema, "monthly": monthly,
        "pv_above_load_points": int((pv > load).sum()), "pv_above_load_days": int((pv > load).any(axis=1).sum()),
        "all_zero_pv_days": int((pv == 0).all(axis=1).sum()),
        "all_zero_pv_forecast_batches": int((forecasts == 0).all(axis=1).sum()),
        "identical_day_rows_extra": {name: int(len(a) - len(np.unique(a, axis=0)))
                                     for name, a in (("load", load), ("pv", pv), ("price", price))},
        "forecast_date_labels_filled": date_fill,
        "forecast_alignment": {"method": "原始整点标签直接对齐实际值；未插值；各批预报分别保留", "matched": len(observations),
                               "outside_actual_range": len(missing_target), "outside_samples": missing_target[:2],
                               "all": error_stats(observations),
                               "positive_actual": error_stats([r for r in observations if r["actual"] > 0]),
                               "zero_actual": error_stats([r for r in observations if r["actual"] == 0]),
                               "zero_nonzero_disagreement_n": len(nonzero_daylight_mismatch),
                               "zero_nonzero_disagreement_samples": nonzero_daylight_mismatch[:5],
                               "lead_bands": {f"{lo}-{hi}": error_stats([r for r in observations if lo <= r["lead"] <= hi and r["actual"] > 0])
                                              for lo, hi in ((1, 6), (7, 12), (13, 18), (19, 24))},
                               "same_target_positive_pairs": {"n": len(pairs),
                                                              "nearer_mae_kw": float(pair_a[:, 0].mean()),
                                                              "farther_mae_kw": float(pair_a[:, 1].mean()),
                                                              "nearer_better_fraction": float((pair_a[:, 0] < pair_a[:, 1]).mean())}},
        "no_changes": ["未删除记录", "未替换零值或极值", "未平滑", "未归一化", "未选择区间积分方式", "未填写模板"]}
    summary["baseline_vs_annual_label_mean"] = {
        name: {"max_abs_difference": float(np.abs(baseline[:, idx] - a.mean(axis=0)).max()),
               "same_after_rounding_4_decimals": int((np.round(baseline[:, idx], 4) == np.round(a.mean(axis=0), 4)).sum())}
        for name, idx, a in (("price", 0, price), ("load", 1, load), ("pv", 2, pv))}
    out = ROOT / "reports"
    (out / "dataset_profile.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {"hours": (np.arange(1, 145) / 6).tolist(), "baseline": baseline.tolist(), "monthly": monthly,
               "profiles": {name: {"mean": a.mean(axis=0).tolist(), "p10": np.quantile(a, .1, axis=0).tolist(),
                                   "p90": np.quantile(a, .9, axis=0).tolist()} for name, a in (("load", load), ("pv", pv), ("price", price))},
               "selected_dates": {d: {"load": load[EXPECTED_DATES.index(d)].tolist(), "pv": pv[EXPECTED_DATES.index(d)].tolist(),
                                      "price": price[EXPECTED_DATES.index(d)].tolist()} for d in ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")}}
    (out / "plot_payload.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
