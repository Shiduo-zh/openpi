import os
import numpy as np
import json
import argparse
import pandas as pd

metric_map = {
    "success_rate": "SR",
    "progress_score": "PS",
    "intention_score": "IS",
}

def visulize_step_bar(df, outfile):
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(df.index, pd.MultiIndex):
        df_plot = df.reset_index()
    else:
        df_plot = df

    metric_alphas = [1.0, 0.50, 0.20]  # SR最深，PS中，IS最浅
    tracks = sorted(df_plot['track'].unique())
    models = sorted(df_plot['model'].unique())
    x = np.arange(len(tracks))
    n_models = len(models)
    total_width = 0.8
    bar_width = total_width / n_models
    color_palette = plt.get_cmap('tab10').colors

    plt.figure(figsize=(max(7, 1.8*len(tracks)), 6))
    ax = plt.gca()

    for m_idx, model in enumerate(models):
        color = color_palette[m_idx % len(color_palette)]
        bar_positions = x - total_width/2 + m_idx * bar_width + bar_width/2

        sr_vals, ps_vals, is_vals = [], [], []
        for track in tracks:
            _row = df_plot[(df_plot['track'] == track) & (df_plot['model'] == model)]
            sr = _row["avg_SR"].values[0] if len(_row["avg_SR"]) else 0.
            ps = _row["avg_PS"].values[0] if len(_row["avg_PS"]) else 0.
            is_ = _row["avg_IS"].values[0] if len(_row["avg_IS"]) else 0.
            sr_vals.append(sr)
            ps_vals.append(ps)
            is_vals.append(is_)

        ax.bar(
            bar_positions, is_vals, width=bar_width,
            color=color, alpha=metric_alphas[2], label=model,
            edgecolor='black', linewidth=0.5
        )
        ax.bar(
            bar_positions, [ps - sr for ps, sr in zip(ps_vals, sr_vals)], width=bar_width,
            bottom=sr_vals,
            color=color, alpha=metric_alphas[1],
            edgecolor='black', linewidth=0.5, label=None
        )
        ax.bar(
            bar_positions, sr_vals, width=bar_width,
            color=color, alpha=metric_alphas[0],
            edgecolor='black', linewidth=0.5, label=None
        )

    ax.set_xlabel('Track')
    ax.set_ylabel('Score')
    ax.set_title("IS/PS/SR")
    ax.set_xticks(x)
    ax.set_xticklabels(tracks, rotation=30)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outfile, dpi=180)
    print(f"图表已保存: {outfile}")


def visulize(df, outfile):
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(df.index, pd.MultiIndex):
        df_plot = df.reset_index()
    else:
        df_plot = df
    metrics = ["avg_SR", "avg_PS", "avg_IS"]
    metric_names = {"avg_SR": "SR", "avg_PS": "PS", "avg_IS": "IS"}
    tracks = sorted(df_plot['track'].unique())
    models = sorted(df_plot['model'].unique())
    x = np.arange(len(tracks))

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    total_width = 0.8
    width = total_width / len(models)

    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx]
        for i, model in enumerate(models):
            values = []
            for track in tracks:
                v = df_plot[(df_plot['track'] == track) & (df_plot['model'] == model)][metric]
                v = v.values[0] if len(v) else np.nan
                values.append(v)
            pos = x + (i - (len(models)-1)/2) * width  # 居中方式
            ax.bar(pos, values, width=width, label=model)
        ax.set_xlabel('Track')
        ax.set_ylabel('Score')
        ax.set_title(f"{metric_names[metric]} on each track")
        ax.set_xticks(x)
        ax.set_xticklabels(tracks, rotation=30)
        if m_idx == n_metrics - 1:
            ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=180)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default="evaluate_results", type=str)
    parser.add_argument('--outfile', default="evaluate_results/vlabench_jax.xlsx", type=str)
    parser.add_argument('--figure', default="evaluate_results/vlabench_jax_figure.png", type=str)
    parser.add_argument('--figure_stack', default="evaluate_results/vlabench_jax_figure_stack.png", type=str)
    args = parser.parse_args()
    root_dir = args.root_dir
    records = []
    task_list = set()
    for exp in os.listdir(root_dir):
        if 'vlabench' not in exp:
            continue
        if not os.path.isdir(os.path.join(root_dir, exp)):
            continue
        for model in os.listdir(os.path.join(root_dir, exp)):
            model_dir = os.path.join(root_dir, exp, model)
            if not os.path.isdir(model_dir):
                continue
            for track in os.listdir(model_dir):
                metric_file = os.path.join(model_dir, track, "metrics.json")
                if os.path.isfile(metric_file):
                    with open(metric_file, "r") as f:
                        metric_data = json.load(f)
                    row = {"model": f"{exp}_{model}", "track": track}
                    sum_success, sum_intention, sum_progress, n = 0, 0, 0, 0
                    for task, scores in metric_data.items():
                        task_list.add(task)
                        for k in ["success_rate", "progress_score", "intention_score"]:
                            value = scores.get(k, None)
                            if value is not None:
                                try:
                                    value = round(float(value), 3)
                                except Exception:
                                    pass
                            row[f"{task}_{metric_map[k]}"] = value
                        sum_success += scores.get("success_rate", 0)
                        sum_intention += scores.get("intention_score", 0)
                        sum_progress += scores.get("progress_score", 0)
                        n += 1
                    # 三列平均分缩写
                    row["avg_SR"] = round(sum_success / n, 3) if n > 0 else None
                    row["avg_IS"] = round(sum_intention / n, 3) if n > 0 else None
                    row["avg_PS"] = round(sum_progress / n, 3) if n > 0 else None
                    records.append(row)
    task_list = sorted(task_list)
    columns = ["model", "track"]
    tuples = [("模型名", ""), ("评测track", "")]

    for task in task_list:
        for k in ["success_rate", "progress_score", "intention_score"]:
            col = f"{task}_{metric_map[k]}"
            columns.append(col)
            tuples.append((task, metric_map[k]))

    columns += ["avg_SR", "avg_IS", "avg_PS"]
    tuples += [("平均分", "SR"), ("平均分", "IS"), ("平均分", "PS")]

    df = pd.DataFrame(records)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    df = df.sort_values(by=["model", "track"])
    df.index = pd.MultiIndex.from_arrays([df['model'], df['track']])
    df = df.drop(columns=['model', 'track'])
    visulize(df, args.figure)
    visulize_step_bar(df, args.figure_stack)
    assert len(df.columns) == len(tuples) - 2  # 注意你df已drop model/track
    df.columns = pd.MultiIndex.from_tuples(tuples[2:])  # 跳过model、track这两个单头
    writer = pd.ExcelWriter(args.outfile, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='summary')
    writer.close()
    print("已保存为 'result.xlsx'，用Excel打开可见合并单元格效果。")
    
    