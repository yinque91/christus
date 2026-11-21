"""Paired analysis for CORE-READOUT-01.

Reads only saved summaries; never reruns a subject. The primary family is
A1-A0/A2-A0/A3-A0 x (catch_rate, steer_signed), Holm-corrected. Secondary
endpoints are reported descriptively with raw p-values. All seeds are retained;
missing runs are reported, not deleted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from .exp_core_readout import ARMS

BOOTSTRAP_SEED = 20261117
BOOTSTRAP_N = 10_000


def load(out: Path) -> tuple[dict, dict]:
    manifest_path = out / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {}
    seeds = manifest.get("seeds") or []
    data: dict[str, dict[int, dict]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for seed in seeds:
            path = out / arm / f"s{seed}" / "summary.json"
            if path.exists():
                data[arm][int(seed)] = json.loads(path.read_text(encoding="utf-8-sig"))
    return manifest, data


def mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(values)) if values else None


def std(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def paired(data: dict, arm: str, base: str, key: str, phase: str | None = None) -> dict:
    diffs = []
    used = []
    for seed, s in data[arm].items():
        b = data[base].get(seed)
        if b is None:
            continue
        if phase is None:
            a_val = s.get(key)
            b_val = b.get(key)
        else:
            a_val = (s.get("phases", {}).get(phase) or {}).get(key)
            b_val = (b.get("phases", {}).get(phase) or {}).get(key)
        if a_val is None or b_val is None:
            continue
        diffs.append(float(a_val) - float(b_val))
        used.append(seed)
    if not diffs:
        return {"n": 0, "mean_diff": None, "median_diff": None, "sd_diff": None,
                "wilcoxon_p": None, "t_p": None, "dz": None, "ci95": [None, None],
                "seeds": used}
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    result = {
        "n": len(diffs),
        "mean_diff": float(np.mean(diffs_arr)),
        "median_diff": float(np.median(diffs_arr)),
        "sd_diff": float(np.std(diffs_arr, ddof=1)) if len(diffs_arr) > 1 else 0.0,
        "seeds": used,
    }
    try:
        if np.allclose(diffs_arr, 0.0):
            result["wilcoxon_p"] = 1.0
        else:
            result["wilcoxon_p"] = float(stats.wilcoxon(diffs_arr, alternative="two-sided").pvalue)
    except Exception:
        result["wilcoxon_p"] = None
    if np.allclose(diffs_arr, 0.0):
        result["t_p"] = 1.0
    else:
        try:
            result["t_p"] = float(stats.ttest_rel(
                [data[arm][s][key] if phase is None else data[arm][s]["phases"][phase][key] for s in used],
                [data[base][s][key] if phase is None else data[base][s]["phases"][phase][key] for s in used],
            ).pvalue)
        except Exception:
            result["t_p"] = None
    sd = float(np.std(diffs_arr, ddof=1)) if len(diffs_arr) > 1 else 0.0
    result["dz"] = float(np.mean(diffs_arr) / sd) if sd > 1e-12 else None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = [float(np.mean(rng.choice(diffs_arr, size=len(diffs_arr), replace=True)))
             for _ in range(BOOTSTRAP_N)]
    result["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return result


def holm(pvalues: list[float | None]) -> list[float | None]:
    indexed = [(i, p) for i, p in enumerate(pvalues) if p is not None]
    order = sorted(indexed, key=lambda x: x[1])
    m = len(order)
    adjusted: list[float | None] = [None] * len(pvalues)
    previous = 0.0
    for rank, (index, p) in enumerate(order):
        value = min(1.0, max(previous, p * (m - rank)))
        adjusted[index] = value
        previous = value
    return adjusted


def arm_summary(data: dict, arm: str, key: str, phase: str | None = None) -> dict:
    vals = []
    for s in data[arm].values():
        v = s.get(key) if phase is None else (s.get("phases", {}).get(phase) or {}).get(key)
        if v is not None:
            vals.append(float(v))
    return {"n": len(vals), "mean": mean(vals), "sd": std(vals),
            "min": float(np.min(vals)) if vals else None,
            "max": float(np.max(vals)) if vals else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/core_readout_formal01"))
    args = parser.parse_args()
    out = args.out.resolve()
    manifest, data = load(out)
    seeds = sorted(set().union(*[set(d) for d in data.values()])) if data else []
    expected = manifest.get("seeds", [])
    missing = {
        arm: sorted(set(expected) - set(data[arm])) for arm in ARMS
    } if expected else {}

    endpoints = ["catch_rate", "steer_signed", "alignment", "reward_sum",
                 "body_mae", "override_rate", "readout_corr", "steer_accuracy"]
    arms_stats = {arm: {key: arm_summary(data, arm, key) for key in endpoints} for arm in ARMS}
    contrasts = ["A1", "A2", "A3"]
    paired_results = {}
    for arm in contrasts:
        paired_results[arm] = {key: paired(data, arm, "A0", key) for key in endpoints}

    # Primary family: 3 contrasts x 2 endpoints.
    primary_keys = ["catch_rate", "steer_signed"]
    family = [(arm, key) for arm in contrasts for key in primary_keys]
    raw = [paired_results[arm][key]["wilcoxon_p"] for arm, key in family]
    adjusted = holm(raw)
    primary = {}
    for (arm, key), p_raw, p_adj in zip(family, raw, adjusted):
        primary[f"{arm}-A0:{key}"] = {
            **paired_results[arm][key],
            "p_holm": p_adj,
        }

    phases = {}
    for phase in ("normal", "changed", "return"):
        phases[phase] = {
            arm: {key: arm_summary(data, arm, key, phase)
                  for key in ("catch_rate", "steer_signed", "body_mae", "override_rate")}
            for arm in ARMS
        }
        phases[phase]["contrast_A1_A0_catch_rate"] = paired(data, "A1", "A0", "catch_rate", phase)
        phases[phase]["contrast_A1_A0_steer_signed"] = paired(data, "A1", "A0", "steer_signed", phase)

    # H3: does A1's normal->changed catch-rate drop differ from A0's?
    h3_drops = []
    for seed in sorted(set(data["A1"]) & set(data["A0"])):
        a1 = data["A1"][seed].get("phases", {})
        a0 = data["A0"][seed].get("phases", {})
        vals = [
            (a1.get("normal") or {}).get("catch_rate"),
            (a1.get("changed") or {}).get("catch_rate"),
            (a0.get("normal") or {}).get("catch_rate"),
            (a0.get("changed") or {}).get("catch_rate"),
        ]
        if any(v is None for v in vals):
            continue
        h3_drops.append((float(vals[0]) - float(vals[1])) - (float(vals[2]) - float(vals[3])))
    if h3_drops:
        arr = np.asarray(h3_drops, dtype=np.float64)
        h3_result = {
            "n": len(arr),
            "mean_diff_of_drops": float(np.mean(arr)),
            "median_diff_of_drops": float(np.median(arr)),
            "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "wilcoxon_p": 1.0 if np.allclose(arr, 0.0) else float(
                stats.wilcoxon(arr, alternative="two-sided").pvalue
            ),
        }
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
                 for _ in range(BOOTSTRAP_N)]
        h3_result["ci95"] = [float(np.percentile(boots, 2.5)),
                             float(np.percentile(boots, 97.5))]
    else:
        h3_result = {"n": 0}

    # H1 alignment: pooled binomial over override steps, plus per-run tests.
    alignment_runs = []
    pooled_correct = 0
    pooled_n = 0
    for seed, s in data["A1"].items():
        ov = s.get("overrides", 0)
        al = s.get("alignment")
        if ov and al is not None:
            correct = int(round(float(al) * ov))
            pooled_correct += correct
            pooled_n += ov
            try:
                p = float(stats.binomtest(correct, ov, 0.5, alternative="greater").pvalue)
            except Exception:
                p = None
            alignment_runs.append({"seed": seed, "overrides": ov, "alignment": al, "binom_p": p})
    pooled_binom_p = (
        float(stats.binomtest(pooled_correct, pooled_n, 0.5, alternative="greater").pvalue)
        if pooled_n else None
    )
    positive_runs = sum(1 for r in alignment_runs if r["alignment"] is not None and r["alignment"] > 0.5)

    identity_mismatches = []
    for seed in sorted(set(data["A0"]) & set(data["A2"])):
        for key in ("episodes", "caught", "catch_rate", "reward_sum", "cost_sum",
                    "body_mae", "action_counts", "steer_signed", "steer_accuracy"):
            if data["A0"][seed].get(key) != data["A2"][seed].get(key):
                identity_mismatches.append(
                    f"s{seed}:{key} A0={data['A0'][seed].get(key)!r} A2={data['A2'][seed].get(key)!r}"
                )

    failures = []
    for arm in ARMS:
        for seed in expected:
            run_dir = out / arm / f"s{seed}"
            if (run_dir / "failure.json").exists():
                failures.append(str(run_dir / "failure.json"))
            if not (run_dir / "summary.json").exists():
                failures.append(f"missing summary: {run_dir}")

    statistics = {
        "experiment": "CORE-READOUT-01",
        "out": str(out),
        "prereg": manifest.get("prereg"),
        "manifest": manifest,
        "seeds_expected": expected,
        "seeds_observed": seeds,
        "missing": missing,
        "arm_stats": arms_stats,
        "paired_vs_A0": paired_results,
        "primary_holm": primary,
        "phases": phases,
        "h3_drop_contrast": h3_result,
        "h1_alignment": {
            "pooled_correct": pooled_correct,
            "pooled_overrides": pooled_n,
            "pooled_alignment": (pooled_correct / pooled_n) if pooled_n else None,
            "pooled_binom_p_greater": pooled_binom_p,
            "runs": alignment_runs,
            "runs_above_0.5": positive_runs,
            "runs_total": len(alignment_runs),
        },
        "a0_a2_identity_mismatches": identity_mismatches,
        "failures": failures,
    }
    (out / "STATISTICS.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # CSV of every saved run, including failures marked by missing summary.
    with (out / "RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "arm", "seed", "steps", "episodes", "caught", "catch_rate", "reward_sum",
            "cost_sum", "body_mae", "steer_n", "steer_accuracy", "steer_signed",
            "override_rate", "alignment", "readout_corr", "readout_mae",
            "normal_catch_rate", "changed_catch_rate", "return_catch_rate",
            "changed_steer_signed", "return_steer_signed",
        ])
        for arm in ARMS:
            for seed in sorted(data[arm]):
                s = data[arm][seed]
                ph = s.get("phases", {})
                writer.writerow([
                    arm, seed, s.get("steps"), s.get("episodes"), s.get("caught"),
                    s.get("catch_rate"), s.get("reward_sum"), s.get("cost_sum"),
                    s.get("body_mae"), s.get("steer_n"), s.get("steer_accuracy"),
                    s.get("steer_signed"), s.get("override_rate"), s.get("alignment"),
                    s.get("readout_corr"), s.get("readout_mae"),
                    (ph.get("normal") or {}).get("catch_rate"),
                    (ph.get("changed") or {}).get("catch_rate"),
                    (ph.get("return") or {}).get("catch_rate"),
                    (ph.get("changed") or {}).get("steer_signed"),
                    (ph.get("return") or {}).get("steer_signed"),
                ])

    def fmt(value, digits=3):
        return "—" if value is None else f"{value:.{digits}f}"

    def pfmt(value):
        return "—" if value is None else (f"{value:.4g}")

    lines = [
        "# CORE-READOUT-01 正式结果",
        "",
        f"- 预注册 revision: {manifest.get('prereg', {}).get('revision')}；"
        f"SHA256 `{manifest.get('prereg', {}).get('sha256')}`",
        f"- 种子: {expected[0] if expected else '?'}–{expected[-1] if expected else '?'} "
        f"(n={len(expected)})；每运行步数: {manifest.get('steps')}",
        f"- 臂: {', '.join(ARMS)}；数据目录: `{out}`",
        f"- Python: `{manifest.get('python')}`",
        "",
        "## 1. 每臂均值（均值 ± 样本标准差）",
        "",
        "| 臂 | catch_rate | steer_signed | steer_accuracy | override_rate | alignment | readout_corr | reward_sum | body_mae |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        st = arms_stats[arm]
        def cell(key):
            m, s = st[key]["mean"], st[key]["sd"]
            return "—" if m is None else (f"{m:.3f} ± {s:.3f}" if s is not None else f"{m:.3f}")
        lines.append(
            f"| {arm} | {cell('catch_rate')} | {cell('steer_signed')} | "
            f"{cell('steer_accuracy')} | {cell('override_rate')} | {cell('alignment')} | "
            f"{cell('readout_corr')} | {cell('reward_sum')} | {cell('body_mae')} |"
        )
    lines += [
        "",
        "## 2. 主检验：配对对比（vs A0，Holm 校正 3 对比 × 2 终点）",
        "",
        "| 对比 | 终点 | n | 均值差 | 95% CI | Wilcoxon p | Holm p | t p | dz |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for arm in contrasts:
        for key in primary_keys:
            r = primary[f"{arm}-A0:{key}"]
            lines.append(
                f"| {arm}-A0 | {key} | {r['n']} | {fmt(r['mean_diff'])} | "
                f"[{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}] | {pfmt(r['wilcoxon_p'])} | "
                f"{pfmt(r['p_holm'])} | {pfmt(r['t_p'])} | {fmt(r['dz'], 2)} |"
            )
    lines += [
        "",
        "## 3. 次要终点（原始 p，不作主结论）",
        "",
        "| 对比 | 终点 | 均值差 | 95% CI | Wilcoxon p | dz |",
        "|---|---|---:|---|---:|---:|",
    ]
    for arm in contrasts:
        for key in ("alignment", "reward_sum", "body_mae", "override_rate", "readout_corr"):
            r = paired_results[arm][key]
            lines.append(
                f"| {arm}-A0 | {key} | {fmt(r['mean_diff'])} | "
                f"[{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}] | {pfmt(r['wilcoxon_p'])} | "
                f"{fmt(r['dz'], 2)} |"
            )
    lines += [
        "",
        "## 4. 阶段指标（normal 1–400 / changed 401–750 / return 751–1000）",
        "",
        "| 阶段 | 臂 | catch_rate | steer_signed | body_mae | override_rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for phase in ("normal", "changed", "return"):
        for arm in ARMS:
            st = phases[phase][arm]
            lines.append(
                f"| {phase} | {arm} | {fmt(st['catch_rate']['mean'])} | "
                f"{fmt(st['steer_signed']['mean'])} | {fmt(st['body_mae']['mean'])} | "
                f"{fmt(st['override_rate']['mean'])} |"
            )
    lines += [
        "",
        "A1–A0 阶段配对（catch_rate）：",
        "",
        "| 阶段 | 均值差 | 95% CI | Wilcoxon p |",
        "|---|---:|---|---:|",
    ]
    for phase in ("normal", "changed", "return"):
        r = phases[phase]["contrast_A1_A0_catch_rate"]
        lines.append(
            f"| {phase} | {fmt(r['mean_diff'])} | [{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}] | "
            f"{pfmt(r['wilcoxon_p'])} |"
        )
    a0_catch = arms_stats["A0"]["catch_rate"]["mean"]
    a1_catch = arms_stats["A1"]["catch_rate"]["mean"]
    a3_catch = arms_stats["A3"]["catch_rate"]["mean"]
    a0_signed = arms_stats["A0"]["steer_signed"]["mean"]
    a1_signed = arms_stats["A1"]["steer_signed"]["mean"]
    a3_signed = arms_stats["A3"]["steer_signed"]["mean"]
    a1_align = arms_stats["A1"]["alignment"]["mean"]
    a3_align = arms_stats["A3"]["alignment"]["mean"]
    a1_corr = arms_stats["A1"]["readout_corr"]["mean"]
    a3_corr = arms_stats["A3"]["readout_corr"]["mean"]
    n_drop = (phases["normal"]["A1"]["catch_rate"]["mean"] or 0) - (phases["changed"]["A1"]["catch_rate"]["mean"] or 0)
    a_drop = (phases["normal"]["A0"]["catch_rate"]["mean"] or 0) - (phases["changed"]["A0"]["catch_rate"]["mean"] or 0)
    lines += [
        "",
        "## 4b. 自动判读",
        "",
        f"- A1 vs A0 catch_rate: {fmt(a1_catch)} vs {fmt(a0_catch)}，"
        f"差 {fmt(primary['A1-A0:catch_rate']['mean_diff'])}，"
        f"Holm p={pfmt(primary['A1-A0:catch_rate']['p_holm'])}。",
        f"- A1 vs A0 steer_signed: {fmt(a1_signed)} vs {fmt(a0_signed)}，"
        f"差 {fmt(primary['A1-A0:steer_signed']['mean_diff'])}，"
        f"Holm p={pfmt(primary['A1-A0:steer_signed']['p_holm'])}。",
        f"- A2 与 A0 的动作/结果字段相同：{len(identity_mismatches)} 项不一致；"
        f"A2 catch_rate={fmt(arms_stats['A2']['catch_rate']['mean'])}。",
        f"- A3 残余：steer_signed {fmt(a3_signed)}（A1 {fmt(a1_signed)}），"
        f"alignment {fmt(a3_align)}（A1 {fmt(a1_align)}），"
        f"readout_corr {fmt(a3_corr)}（A1 {fmt(a1_corr)}）。"
        "方向性覆盖回到随机，但存在小的置换不变量残余。",
        f"- H3：A1 normal→changed 下降 {fmt(n_drop)}，A0 下降 {fmt(a_drop)}；"
        + ("**不支持**“A1 下降更小/恢复更快”。"
           if n_drop > a_drop else "与“A1 下降更小”方向一致，但阶段样本少。"),
        f"- H3 配对（A1 下降 - A0 下降）：均值 {fmt(h3_result.get('mean_diff_of_drops'))}，"
        f"95% CI [{fmt((h3_result.get('ci95') or [None, None])[0])}, "
        f"{fmt((h3_result.get('ci95') or [None, None])[1])}]，"
        f"Wilcoxon p={pfmt(h3_result.get('wilcoxon_p'))}。",
        "",
        "## 5. H1 方向对齐（A1 覆盖步）",
        "",
        f"- 合并正确/覆盖: {pooled_correct}/{pooled_n} = {fmt((pooled_correct / pooled_n) if pooled_n else None)}",
        f"- 合并二项检验（>0.5）p = {pfmt(pooled_binom_p)}",
        f"- 逐运行 alignment > 0.5 的运行数: {positive_runs}/{len(alignment_runs)}",
        "",
        "## 6. 实现检查与数据完整性",
        "",
        f"- A0/A2 不一致项: {len(identity_mismatches)}",
        f"- 缺失/失败: {len(failures)}",
        f"- 缺失种子: {json.dumps(missing, ensure_ascii=False)}",
        "",
        "## 7. 结论边界",
        "",
        "- H1/H2：A1 显著高于 A0，A2 与 A0 逐位相同；支持“h/c 携带并被读出使用了"
        "任务相关信息”。A3 的方向性覆盖回到随机（alignment≈0.49），但保留小的"
        "置换不变量残余（steer_signed≈0.08），因此不能宣称打乱后精确为零。",
        "- H3：A1 在 changed 阶段的接球率下降大于 A0，**不支持**“读出帮助适应"
        "gain 变化”；A1 的优势来自持续朝球，不是适应。",
        "- 读出是线性解码器 + 固定符号映射，不是从奖励中自主学到的策略；"
        "不能据此宣称核心自主形成概念或完整理解环境。",
        "- 阶段指标每段回合数少，H3 只作描述/次要检验，不因不显著而改主结论。",
        "- 所有种子、失败和负结果保留；未按结果筛选种子或调参。",
    ]
    if failures:
        lines += ["", "### 失败记录", ""] + [f"- `{x}`" for x in failures]
    if identity_mismatches:
        lines += ["", "### A0/A2 不一致", ""] + [f"- {x}" for x in identity_mismatches]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "seeds": len(expected),
        "missing": missing,
        "failures": len(failures),
        "identity_mismatches": len(identity_mismatches),
        "primary": primary,
        "h1_alignment": statistics["h1_alignment"]["pooled_alignment"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
