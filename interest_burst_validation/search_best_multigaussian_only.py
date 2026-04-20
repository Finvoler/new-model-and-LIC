import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class GMMFitResult:
    user_id: int
    category_id: int
    sample_size: int
    n_components: int
    mse: float
    r2: float
    aic: float
    bic: float


def gaussian_mixture(x: np.ndarray, *params: float) -> np.ndarray:
    """多峰高斯和：sum_i amp_i * exp(-(x-mu_i)^2/(2*sigma_i^2))"""
    y = np.zeros_like(x, dtype=float)
    n_components = len(params) // 3
    for i in range(n_components):
        amp = params[3 * i]
        mu = params[3 * i + 1]
        sigma = max(params[3 * i + 2], 1e-6)
        y += amp * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
    return y


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def calc_aic_bic(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> Tuple[float, float]:
    n = len(y_true)
    rss = float(np.sum((y_true - y_pred) ** 2))
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * n_params
    bic = n * np.log(rss / n) + n_params * np.log(n)
    return float(aic), float(bic)


def load_pv_data(csv_path: str) -> pd.DataFrame:
    cols = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
    dtypes = {
        "user_id": "int64",
        "item_id": "int64",
        "category_id": "int64",
        "behavior_type": "category",
        "timestamp": "int64",
    }
    df = pd.read_csv(csv_path, header=None, names=cols, dtype=dtypes)
    df = df[df["behavior_type"] == "pv"].copy()

    dt = pd.to_datetime(df["timestamp"], unit="s")
    df["hour_in_day"] = dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0

    return df[["user_id", "category_id", "hour_in_day"]]


def build_hist(hours: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    hist_density, edges = np.histogram(hours, bins=bins, range=(0.0, 24.0), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, hist_density


def init_params(x: np.ndarray, y: np.ndarray, n_components: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    peak_idx = np.argsort(y)[::-1][:n_components]
    peak_idx = np.sort(peak_idx)

    p0 = []
    lower = []
    upper = []

    y_max = float(np.max(y)) if len(y) > 0 else 1.0
    for idx in peak_idx:
        amp0 = float(max(y[idx], 0.3 * y_max))
        mu0 = float(x[idx])
        sigma0 = 1.0

        p0.extend([amp0, mu0, sigma0])
        lower.extend([0.0, 0.0, 0.15])
        upper.extend([np.inf, 24.0, 8.0])

    return np.array(p0, dtype=float), (np.array(lower, dtype=float), np.array(upper, dtype=float))


def fit_given_components(x: np.ndarray, y: np.ndarray, n_components: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    p0, bounds = init_params(x, y, n_components)
    try:
        params, _ = curve_fit(
            gaussian_mixture,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=120000,
        )
    except Exception:
        return None

    y_pred = gaussian_mixture(x, *params)
    return params, y_pred


def fit_best_gmm(
    x: np.ndarray,
    y: np.ndarray,
    min_components: int,
    max_components: int,
    model_select: str,
) -> Optional[Tuple[int, np.ndarray, np.ndarray, float, float, float, float]]:
    best = None

    for c in range(min_components, max_components + 1):
        result = fit_given_components(x, y, c)
        if result is None:
            continue
        params, y_pred = result

        cur_mse = mse(y, y_pred)
        cur_r2 = r2_score_manual(y, y_pred)
        cur_aic, cur_bic = calc_aic_bic(y, y_pred, n_params=3 * c)

        if model_select == "aic":
            score = cur_aic
        elif model_select == "bic":
            score = cur_bic
        else:
            score = -cur_r2

        if (best is None) or (score < best[0]):
            best = (score, c, params, y_pred, cur_mse, cur_r2, cur_aic, cur_bic)

    if best is None:
        return None

    _, c, params, y_pred, cur_mse, cur_r2, cur_aic, cur_bic = best
    return c, params, y_pred, cur_mse, cur_r2, cur_aic, cur_bic


def sanitize_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def evaluate_pair(
    pair_df: pd.DataFrame,
    user_id: int,
    category_id: int,
    bins: int,
    min_components: int,
    max_components: int,
    model_select: str,
) -> Optional[GMMFitResult]:
    hours = pair_df["hour_in_day"].to_numpy(dtype=float)
    if len(hours) < max(120, bins):
        return None

    x, y = build_hist(hours, bins)
    fit = fit_best_gmm(x, y, min_components, max_components, model_select)
    if fit is None:
        return None

    n_components, _, y_pred, cur_mse, cur_r2, cur_aic, cur_bic = fit

    return GMMFitResult(
        user_id=int(user_id),
        category_id=int(category_id),
        sample_size=int(len(hours)),
        n_components=int(n_components),
        mse=float(cur_mse),
        r2=float(cur_r2),
        aic=float(cur_aic),
        bic=float(cur_bic),
    )


# --- 科研论文绘图全局配置 ---
_RESEARCH_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.fontsize': 11,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.6',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
}
_COMP_COLORS = ['#EE7733', '#009988', '#0077BB', '#EE3377', '#33BBEE', '#BBBBBB']


def plot_pair(
    pair_df: pd.DataFrame,
    fit_result: GMMFitResult,
    bins: int,
    min_components: int,
    max_components: int,
    model_select: str,
    output_dir: str,
) -> None:
    hours = pair_df["hour_in_day"].to_numpy(dtype=float)
    x, y = build_hist(hours, bins)

    fit = fit_best_gmm(x, y, min_components, max_components, model_select)
    if fit is None:
        return

    n_components, params, _, cur_mse, cur_r2, _, _ = fit

    x_plot = np.linspace(0.0, 24.0, 800)
    y_total = gaussian_mixture(x_plot, *params)

    with plt.rc_context(_RESEARCH_RC):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(
            hours,
            bins=bins,
            range=(0, 24),
            density=True,
            alpha=0.30,
            color='#88CCEE',
            edgecolor='white',
            linewidth=0.4,
            label="Observed histogram",
        )

        ax.plot(
            x_plot,
            y_total,
            color='#CC3311',
            linewidth=2.8,
            label=f"GMM ({n_components} peaks)  MSE={cur_mse:.5f}  $R^2$={cur_r2:.4f}",
            zorder=5,
        )

        for i in range(n_components):
            amp = params[3 * i]
            mu = params[3 * i + 1]
            sigma = max(params[3 * i + 2], 1e-6)
            y_comp = amp * np.exp(-((x_plot - mu) ** 2) / (2.0 * sigma**2))
            ax.plot(
                x_plot, y_comp,
                linestyle='--', linewidth=1.4, alpha=0.7,
                color=_COMP_COLORS[i % len(_COMP_COLORS)],
                label=f"Peak {i+1} ($\\mu$={mu:.1f}h, $\\sigma$={sigma:.2f})",
            )

        ax.set_title(
            f"User {fit_result.user_id}  |  Category {fit_result.category_id}  |  $n$={fit_result.sample_size}",
            fontweight='bold', pad=10,
        )
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Probability Density")
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.tick_params(which='both', top=True, right=True)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()

        name = sanitize_filename(
            f"gmm_user_{fit_result.user_id}_cat_{fit_result.category_id}_n_{fit_result.sample_size}_k_{n_components}"
        )
        fig.savefig(os.path.join(output_dir, f"{name}.png"))
        fig.savefig(os.path.join(output_dir, f"{name}.pdf"))
        plt.close(fig)

def search_best_pairs(
    df: pd.DataFrame,
    min_interactions: int,
    target_count: int,
    bins: int,
    min_components: int,
    max_components: int,
    model_select: str,
    r2_threshold: float,
    max_candidates: int,
) -> List[GMMFitResult]:
    pair_counts = (
        df.groupby(["user_id", "category_id"], as_index=False)
        .size()
        .rename(columns={"size": "interaction_count"})
        .sort_values("interaction_count", ascending=False)
    )

    valid = pair_counts[pair_counts["interaction_count"] >= min_interactions].copy()
    if max_candidates > 0:
        valid = valid.head(max_candidates)

    if valid.empty:
        raise ValueError("没有满足最小交互阈值的(user_id, category_id)样本。")

    selected_data = df.merge(valid[["user_id", "category_id"]], on=["user_id", "category_id"], how="inner")

    picked: List[GMMFitResult] = []
    for row in valid.itertuples(index=False):
        u = int(row.user_id)
        c = int(row.category_id)

        pair_df = selected_data[(selected_data["user_id"] == u) & (selected_data["category_id"] == c)]
        result = evaluate_pair(
            pair_df=pair_df,
            user_id=u,
            category_id=c,
            bins=bins,
            min_components=min_components,
            max_components=max_components,
            model_select=model_select,
        )
        if result is None:
            continue

        if result.r2 >= r2_threshold:
            picked.append(result)
            print(
                f"找到第{len(picked)}组: user={u}, cat={c}, n={result.sample_size}, "
                f"K={result.n_components}, R2={result.r2:.4f}, MSE={result.mse:.6f}"
            )

        if len(picked) >= target_count:
            break

    if len(picked) < target_count:
        # 阈值不足时回退：在所有已拟合中按R2补足
        all_results: List[GMMFitResult] = []
        for row in valid.itertuples(index=False):
            u = int(row.user_id)
            c = int(row.category_id)
            pair_df = selected_data[(selected_data["user_id"] == u) & (selected_data["category_id"] == c)]
            result = evaluate_pair(
                pair_df=pair_df,
                user_id=u,
                category_id=c,
                bins=bins,
                min_components=min_components,
                max_components=max_components,
                model_select=model_select,
            )
            if result is not None:
                all_results.append(result)

        all_results = sorted(all_results, key=lambda t: (t.r2, -t.mse), reverse=True)
        have = {(r.user_id, r.category_id) for r in picked}
        for r in all_results:
            key = (r.user_id, r.category_id)
            if key in have:
                continue
            picked.append(r)
            have.add(key)
            if len(picked) >= target_count:
                break

    if not picked:
        raise RuntimeError("没有找到可用拟合样本，请放宽参数。")

    picked = sorted(picked, key=lambda t: (t.r2, -t.mse), reverse=True)[:target_count]
    return picked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="仅用多峰高斯在淘宝数据中搜索5组高质量兴趣爆发样本")
    parser.add_argument("--csv_path", type=str, default="UserBehavior.csv")
    parser.add_argument("--output_dir", type=str, default="interest_burst_validation/outputs_multi_gaussian_only")
    parser.add_argument("--min_interactions", type=int, default=100)
    parser.add_argument("--target_count", type=int, default=5)
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument("--min_components", type=int, default=2)
    parser.add_argument("--max_components", type=int, default=6)
    parser.add_argument("--model_select", type=str, default="bic", choices=["aic", "bic", "r2"])
    parser.add_argument("--r2_threshold", type=float, default=0.90)
    parser.add_argument("--max_candidates", type=int, default=6000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/4] 加载并过滤 pv 数据...")
    df = load_pv_data(args.csv_path)
    print(f"    pv 总记录: {len(df):,}")

    print("[2/4] 搜索高质量多峰高斯样本...")
    best_results = search_best_pairs(
        df=df,
        min_interactions=args.min_interactions,
        target_count=args.target_count,
        bins=args.bins,
        min_components=args.min_components,
        max_components=args.max_components,
        model_select=args.model_select,
        r2_threshold=args.r2_threshold,
        max_candidates=args.max_candidates,
    )

    print("[3/4] 生成可视化图...")
    selected_keys = pd.DataFrame(
        [{"user_id": r.user_id, "category_id": r.category_id} for r in best_results]
    )
    selected_data = df.merge(selected_keys, on=["user_id", "category_id"], how="inner")

    for r in best_results:
        pair_df = selected_data[(selected_data["user_id"] == r.user_id) & (selected_data["category_id"] == r.category_id)]
        plot_pair(
            pair_df=pair_df,
            fit_result=r,
            bins=args.bins,
            min_components=args.min_components,
            max_components=args.max_components,
            model_select=args.model_select,
            output_dir=args.output_dir,
        )

    print("[4/4] 保存结果表...")
    out_df = pd.DataFrame([r.__dict__ for r in best_results])
    out_df = out_df.sort_values(["r2", "mse"], ascending=[False, True])
    out_csv = os.path.join(args.output_dir, "best5_multi_gaussian_only.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(out_df.to_string(index=False))
    print(f"\n结果表: {out_csv}")
    print(f"图像目录: {args.output_dir}")


if __name__ == "__main__":
    main()
