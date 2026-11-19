"""Saved-sums-only comparison: original FuXi versus MAIN zero-ZHD/ZWD rollout."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from evaluate import CHANNELS, channel_unit
from data.rollout_identity import file_identity


def percentage(original, adapted):
    return 100. * (original - adapted) / original if original > 1e-12 else None


def baseline_tables(sums):
    if sums.shape != (8, 7, 70) or not np.isfinite(sums).all() or (sums[:, 4] <= 0).any():
        raise ValueError("invalid eight-lead sums or missing channel targets")
    rows, leads = [], []
    for j in range(8):
        values = sums[j, :4] / sums[j, 4]
        halo = np.divide(sums[j, 5], sums[j, 6], out=np.full(70, np.nan), where=sums[j, 6] > 0)
        leads.append({"lead_hours":6*(j+1),
            "global_latitude_weighted_mae_normalized":float(sums[j, 0].sum()/sums[j, 4].sum()),
            "halo_latitude_weighted_mae_normalized":float(sums[j, 5].sum()/sums[j, 6].sum())
                if sums[j, 6].sum() > 0 else None,
            "persistence_latitude_weighted_mae_normalized":float(sums[j, 2].sum()/sums[j, 4].sum())})
        for i, name in enumerate(CHANNELS):
            rows.append({"lead_hours":6*(j+1), "channel":name, "unit":channel_unit(name),
                "global_mae_normalized":float(values[0, i]), "global_mae_physical":float(values[1, i]),
                "persistence_mae_normalized":float(values[2, i]), "persistence_mae_physical":float(values[3, i]),
                "halo_mae_normalized":float(halo[i]) if np.isfinite(halo[i]) else None})
    return rows, leads


def comparison_tables(original, reference, leads, reference_summary):
    keyed = {(int(r["lead_hours"]), r["channel"]):r for r in reference}
    expected = {(6*j, c) for j in range(1, 9) for c in CHANNELS}
    if set(keyed) != expected or len(reference) != len(expected):
        raise ValueError("reference must have exactly 70 channels x 8 leads")
    rows = []
    for a in original:
        b = keyed[(a["lead_hours"], a["channel"])]
        if a["unit"] != b["unit"]:
            raise ValueError("physical units differ")
        # A prediction-independent check against the actual saved evaluation:
        # identical targets, weighting, inverse transforms and sampled origins.
        for key in ("persistence_mae_normalized", "persistence_mae_physical"):
            if not np.isclose(a[key], float(b[key]), rtol=2e-5, atol=2e-7):
                raise ValueError(f"paired-data check failed at {a['lead_hours']}h/{a['channel']}/{key}: "
                                 f"original-run={a[key]}, reference={b[key]}")
        row = {k:a[k] for k in ("lead_hours", "channel", "unit")}
        for metric in ("global_mae_normalized", "global_mae_physical", "halo_mae_normalized"):
            x, y = a[metric], float(b[metric]) if b[metric] not in (None, "") else None
            row["fuxi_"+metric] = x
            row["zero_"+metric] = y
            row[metric+"_reduction"] = x-y if x is not None and y is not None else None
            row[metric+"_improvement_pct"] = percentage(x, y) if x is not None and y is not None else None
        rows.append(row)
    summaries = []
    ref_leads = {r["lead_hours"]:r for r in reference_summary["leads"]}
    for a in leads:
        h = a["lead_hours"]; b = ref_leads[h]
        row = {"lead_hours":h}
        for region in ("global", "halo"):
            key = region+"_latitude_weighted_mae_normalized"
            x, y = a[key], b[key]
            row["fuxi_"+region] = x; row["zero_"+region] = y
            row[region+"_improvement_pct"] = percentage(x, y) if x is not None and y is not None else None
        row["global_improved_channels"] = sum(
            r["global_mae_normalized_reduction"] > 0 for r in rows if r["lead_hours"] == h)
        summaries.append(row)
    return rows, summaries


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def figures(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_lead = {h:[r for r in rows if r["lead_hours"] == h] for h in range(6, 49, 6)}
    for h, subset in by_lead.items():
        fig, axes = plt.subplots(5, 1, figsize=(16, 15), constrained_layout=True)
        for ax, section in zip(axes, np.array_split(np.arange(70), 5)):
            x = np.arange(len(section)); rr = [subset[i] for i in section]
            ax.bar(x-.2, [r["fuxi_global_mae_normalized"] for r in rr], .4, label="Original FuXi")
            ax.bar(x+.2, [r["zero_global_mae_normalized"] for r in rr], .4, label="Zero ZHD/ZWD (fine-tuned)")
            ax.set_xticks(x, [r["channel"] for r in rr], rotation=40, ha="right")
            ax.set_ylabel("Normalized MAE"); ax.grid(axis="y", alpha=.25)
        axes[0].legend(); fig.suptitle(f"+{h}h | global latitude-weighted MAE | paired full test")
        fig.savefig(output/f"channel_mae_{h:02d}h.png", dpi=150); plt.close(fig)
    matrix = np.array([[r["global_mae_normalized_improvement_pct"] for r in by_lead[h]] for h in by_lead], float)
    limit = max(float(np.nanmax(np.abs(matrix))), .01)
    fig, ax = plt.subplots(figsize=(22, 5), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
    ax.set_xticks(range(70), CHANNELS, rotation=90, fontsize=8)
    ax.set_yticks(range(8), [f"{h}h" for h in by_lead])
    ax.set_title("Zero-input fine-tuned model vs original FuXi: MAE reduction (%) | green = improved")
    fig.colorbar(im, ax=ax, label="100 * (FuXi MAE - zero-input MAE) / FuXi MAE")
    fig.savefig(output/"channel_improvement_all_leads.png", dpi=160); plt.close(fig)
    # Physical units cannot be combined on one shared absolute-error axis.
    subset = by_lead[48]
    groups = [subset[i:i+13] for i in range(0, 65, 13)]
    groups += [[r for r in subset if r["channel"] == name] for name in ("t2m", "u10", "v10", "msl", "tp")]
    fig, axes = plt.subplots(5, 2, figsize=(19, 19), constrained_layout=True)
    for ax, rr in zip(axes.flat, groups):
        x = np.arange(len(rr))
        ax.bar(x-.2, [r["fuxi_global_mae_physical"] for r in rr], .4, label="Original FuXi")
        ax.bar(x+.2, [r["zero_global_mae_physical"] for r in rr], .4, label="Zero ZHD/ZWD")
        ax.set_xticks(x, [r["channel"] for r in rr], rotation=50, ha="right")
        ax.set_ylabel(f"MAE ({rr[0]['unit']})"); ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(); fig.suptitle("+48h | inverse-normalized, global latitude-weighted MAE")
    fig.savefig(output/"channel_mae_physical_48h.png", dpi=150); plt.close(fig)


def make_comparison(output):
    output = Path(output)
    with np.load(output/"sums.npz") as saved:
        meta = json.loads(str(saved["identity"])); sums = saved["sums"]; count = int(saved["samples"])
    if count != meta["samples"]:
        raise ValueError("saved sample count mismatch")
    original, leads = baseline_tables(sums)
    write_csv(output/"original_mae_by_channel.csv", original)
    write_csv(output/"original_mae_by_lead.csv", leads)
    summary = {"evaluation":meta, "samples":count, "leads":leads}
    (output/"original_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    if not meta["full_test"]:
        print("SMOKE SUBSET ONLY: original metrics saved; no comparison against full-test DA results.", flush=True)
        return
    exp = Path(meta["reference_dir"])
    for name, path in (("reference_csv", exp/"test/rollout_mae_by_channel.csv"),
                       ("reference_summary", exp/"test/rollout_mae_summary.json"),
                       ("reference_config", exp/"resolved_config.json")):
        if file_identity(path) != meta[name]:
            raise ValueError(f"reference changed since inference: {name}")
    reference_summary = json.loads((exp/"test/rollout_mae_summary.json").read_text())
    if count != reference_summary["samples"]:
        raise ValueError("full test sample count differs from reference")
    with (exp/"test/rollout_mae_by_channel.csv").open() as f:
        reference = list(csv.DictReader(f))
    rows, summaries = comparison_tables(original, reference, leads, reference_summary)
    write_csv(output/"channel_comparison_all_leads.csv", rows)
    write_csv(output/"summary_by_lead.csv", summaries)
    figures(output, rows)
    lines = ["# 原始 FuXi 与零 ZHD/ZWD 微调模型：逐通道对比", "",
             f"共同测试起报样本：{count}；{meta['origin_first']} 至 {meta['origin_last']}。", "",
             "原始模型为 fuxi_short.pth，所有步骤使用同一套未微调权重，无观测输入。",
             "对照为 main 分支已保存的 stage_48h/best.pt 测试结果；不重新推理该模型。", "",
             "提升率 = 100 ×（原始 FuXi MAE − 零输入模型 MAE）/ 原始 FuXi MAE。正值表示零输入模型更好。",
             "两种单位口径的 persistence 校验已通过。归一化 MAE 用于跨通道汇总；物理 MAE 按通道单位报告。",
             "TP 是各预报时效末尾 6h 的降水量，不是从起报开始累计的 48h 降水。", "",
             "| 时效 | FuXi 全球 MAE | 零输入全球 MAE | 提升率 | 改善通道数 |", "|---|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['lead_hours']}h | {row['fuxi_global']:.6f} | {row['zero_global']:.6f} | "
                     f"{row['global_improvement_pct']:+.3f}% | {row['global_improved_channels']}/70 |")
    lines += ["", "## 48h 逐通道物理 MAE", "", "| 通道 | 单位 | 原始 FuXi | 零输入模型 | 物理 MAE 提升率 |",
              "|---|---|---:|---:|---:|"]
    for row in rows:
        if row["lead_hours"] != 48:
            continue
        p = row["global_mae_physical_improvement_pct"]
        ptext = f"{p:+.3f}%" if p is not None else "NA"
        lines.append(f"| {row['channel']} | {row['unit']} | {row['fuxi_global_mae_physical']:.6g} | "
                     f"{row['zero_global_mae_physical']:.6g} | {ptext} |")
    lines += ["", "本对比是测试集点估计，不代表已通过统计显著性检验；微调模型使用了额外训练，而原始 FuXi 未微调。"]
    (output/"comparison_report.md").write_text("\n".join(lines)+"\n")
    print(f"Full paired comparison saved: {output}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("output_dir", type=Path)
    make_comparison(p.parse_args().output_dir)
