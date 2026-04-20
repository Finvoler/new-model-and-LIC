import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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



@dataclass
class FitResult:
    user_id: int
    category_id: int
    sample_size: int
    mse_gmm: float
    r2_gmm: float
    mse_fourier: float
    r2_fourier: float
    gmm_better: bool


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def gaussian_mixture(x: np.ndarray, *params: float) -> np.ndarray:
    """
    多峰高斯模型：sum_i amp_i * exp(-((x-mu_i)^2)/(2*sigma_i^2))
    参数顺序：[amp1, mu1, sigma1, amp2, mu2, sigma2, ...]
    """
    y = np.zeros_like(x, dtype=float)
    n_components = len(params) // 3
    for i in range(n_components):
        amp = params[3 * i]
        mu = params[3 * i + 1]
        sigma = max(params[3 * i + 2], 1e-6)
        y += amp * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
    return y


def fourier_series(x: np.ndarray, *params: float, order: int) -> np.ndarray:
    w = 2.0 * np.pi / 24.0
    a0 = params[0]
    y = np.full_like(x, a0, dtype=float)
    for k in range(1, order + 1):
        ak = params[2 * k - 1]
        bk = params[2 * k]
        y += ak * np.cos(k * w * x) + bk * np.sin(k * w * x)
    return y


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
    hist_density, bin_edges = np.histogram(hours, bins=bins, range=(0.0, 24.0), density=True)
    x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return x_centers, hist_density


def init_gmm_params(x: np.ndarray, y: np.ndarray, n_components: int) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # 取直方图最高的n个bin作为初始均值
    peak_idx = np.argsort(y)[::-1][:n_components]
    peak_idx = np.sort(peak_idx)

    p0 = []
    lower = []
    upper = []

    y_max = float(np.max(y)) if len(y) > 0 else 1.0
    base_sigma = 1.2

    for idx in peak_idx:
        amp0 = float(max(y[idx], y_max * 0.3))
        mu0 = float(x[idx])
        sigma0 = base_sigma

        p0.extend([amp0, mu0, sigma0])
        lower.extend([0.0, 0.0, 0.2])
        upper.extend([np.inf, 24.0, 12.0])

    return np.array(p0, dtype=float), (np.array(lower, dtype=float), np.array(upper, dtype=float))


def fit_gmm_curve(x: np.ndarray, y: np.ndarray, n_components: int) -> np.ndarray:
    p0, bounds = init_gmm_params(x, y, n_components)
    params, _ = curve_fit(
        gaussian_mixture,
        x,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=100000,
    )
    return params


def fit_fourier_curve(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    p0 = np.zeros(1 + 2 * order, dtype=float)
    p0[0] = float(np.mean(y))

    def _wrapper(xv: np.ndarray, *params: float) -> np.ndarray:
        return fourier_series(xv, *params, order=order)

    params, _ = curve_fit(_wrapper, x, y, p0=p0, maxfev=50000)
    return params


def sanitize_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def evaluate_pair(
    pair_df: pd.DataFrame,
    user_id: int,
    category_id: int,
    bins: int,
    gmm_components: int,
    fourier_order: int,
) -> Optional[FitResult]:
    hours = pair_df["hour_in_day"].to_numpy(dtype=float)
    if len(hours) < max(100, bins):
        return None

    x, y = build_hist(hours, bins)

    try:
        gmm_params = fit_gmm_curve(x, y, gmm_components)
        y_gmm = gaussian_mixture(x, *gmm_params)
    except Exception:
        return None

    try:
        fourier_params = fit_fourier_curve(x, y, fourier_order)
        y_fourier = fourier_series(x, *fourier_params, order=fourier_order)
    except Exception:
        return None

    mse_g = mse(y, y_gmm)
    r2_g = r2_score_manual(y, y_gmm)
    mse_f = mse(y, y_fourier)
    r2_f = r2_score_manual(y, y_fourier)

    return FitResult(
        user_id=int(user_id),
        category_id=int(category_id),
        sample_size=int(len(hours)),
        mse_gmm=mse_g,
        r2_gmm=r2_g,
        mse_fourier=mse_f,
        r2_fourier=r2_f,
        gmm_better=bool(mse_g < mse_f),
    )


def plot_pair(
    pair_df: pd.DataFrame,
    result: FitResult,
    bins: int,
    gmm_components: int,
    fourier_order: int,
    output_dir: str,
) -> None:
    hours = pair_df["hour_in_day"].to_numpy(dtype=float)
    x, y = build_hist(hours, bins)

    gmm_params = fit_gmm_curve(x, y, gmm_components)
    fourier_params = fit_fourier_curve(x, y, fourier_order)

    x_plot = np.linspace(0.0, 24.0, 720)
    y_gmm_plot = gaussian_mixture(x_plot, *gmm_params)
    y_fourier_plot = fourier_series(x_plot, *fourier_params, order=fourier_order)

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
            y_gmm_plot,
            color='#CC3311',
            linewidth=2.8,
            label=f"Gaussian mixture ({gmm_components} peaks)  MSE={result.mse_gmm:.5f}  $R^2$={result.r2_gmm:.4f}",
            zorder=5,
        )
        ax.plot(
            x_plot,
            y_fourier_plot,
            color='#009988',
            linewidth=2.4,
            linestyle='-.',
            label=f"Fourier series (K={fourier_order})  MSE={result.mse_fourier:.5f}  $R^2$={result.r2_fourier:.4f}",
            zorder=4,
        )

        winner = "Gaussian mixture" if result.gmm_better else "Fourier series"
        ax.set_title(
            f"User {result.user_id}  |  Category {result.category_id}  |  $n$={result.sample_size}  |  Better: {winner}",
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
            f"best_user_{result.user_id}_cat_{result.category_id}_n_{result.sample_size}"
        )
        fig.savefig(os.path.join(output_dir, f"{name}.png"))
        fig.savefig(os.path.join(output_dir, f"{name}.pdf"))
        plt.close(fig)

def search_best_pairs(
    df: pd.DataFrame,
    min_interactions: int,
    target_count: int,
    bins: int,
    gmm_components: int,
    fourier_order: int,
    r2_threshold: float,
    max_candidates: int,
) -> Tuple[pd.DataFrame, List[FitResult]]:
    pair_counts = (
        df.groupby(["user_id", "category_id"], as_index=False)
        .size()
        .rename(columns={"size": "interaction_count"})
    )

    valid = pair_counts[pair_counts["interaction_count"] >= min_interactions].copy()
    valid = valid.sort_values("interaction_count", ascending=False)

    if max_candidates > 0:
        valid = valid.head(max_candidates)

    if valid.empty:
        raise ValueError("没有满足最小交互阈值的(user_id, category_id)对。")

    selected_data = df.merge(valid[["user_id", "category_id"]], on=["user_id", "category_id"], how="inner")

    good_results: List[FitResult] = []

    for row in valid.itertuples(index=False):
        user_id = int(row.user_id)
        category_id = int(row.category_id)

        pair_df = selected_data[
            (selected_data["user_id"] == user_id) & (selected_data["category_id"] == category_id)
        ]

        result = evaluate_pair(
            pair_df=pair_df,
            user_id=user_id,
            category_id=category_id,
            bins=bins,
            gmm_components=gmm_components,
            fourier_order=fourier_order,
        )

        if result is None:
            continue

        # 定义“嘎嘎好”：多峰高斯R2达到阈值，且不明显劣于傅里叶
        fourier_safe_floor = result.mse_fourier * 1.20
        is_good = (result.r2_gmm >= r2_threshold) and (result.mse_gmm <= fourier_safe_floor)

        if is_good:
            good_results.append(result)
            print(
                f"找到第{len(good_results)}组: user={user_id}, cat={category_id}, "
                f"n={result.sample_size}, R2_gmm={result.r2_gmm:.4f}, R2_fourier={result.r2_fourier:.4f}"
            )

        if len(good_results) >= target_count:
            break

    # 若高阈值不足5组，则按R2_gmm排序补足
    if len(good_results) < target_count:
        print("高标准样本不足，开始按多峰高斯R2全局补足到目标数量...")
        all_results: List[FitResult] = []

        for row in valid.itertuples(index=False):
            user_id = int(row.user_id)
            category_id = int(row.category_id)
            pair_df = selected_data[
                (selected_data["user_id"] == user_id) & (selected_data["category_id"] == category_id)
            ]
            result = evaluate_pair(
                pair_df=pair_df,
                user_id=user_id,
                category_id=category_id,
                bins=bins,
                gmm_components=gmm_components,
                fourier_order=fourier_order,
            )
            if result is not None:
                all_results.append(result)

        all_results = sorted(all_results, key=lambda x: (x.r2_gmm, -x.mse_gmm), reverse=True)

        existing = {(r.user_id, r.category_id) for r in good_results}
        for r in all_results:
            key = (r.user_id, r.category_id)
            if key in existing:
                continue
            good_results.append(r)
            existing.add(key)
            if len(good_results) >= target_count:
                break

    if not good_results:
        raise RuntimeError("未能找到可用拟合样本，请调整参数。")

    good_results = sorted(good_results, key=lambda x: (x.r2_gmm, -x.mse_gmm), reverse=True)[:target_count]

    good_df = pd.DataFrame([r.__dict__ for r in good_results])
    return valid, good_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="搜索并可视化多峰高斯拟合效果最好的5组淘宝兴趣爆发样本")
    parser.add_argument("--csv_path", type=str, default="UserBehavior.csv")
    parser.add_argument("--output_dir", type=str, default="interest_burst_validation/outputs_multi_gaussian")
    parser.add_argument("--min_interactions", type=int, default=100)
    parser.add_argument("--target_count", type=int, default=5)
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument("--gmm_components", type=int, default=3)
    parser.add_argument("--fourier_order", type=int, default=5)
    parser.add_argument("--r2_threshold", type=float, default=0.80)
    parser.add_argument("--max_candidates", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/5] 加载pv数据...")
    df = load_pv_data(args.csv_path)
    print(f"    pv总记录: {len(df):,}")

    print("[2/5] 搜索多峰高斯拟合优质样本...")
    valid_pairs, best_results = search_best_pairs(
        df=df,
        min_interactions=args.min_interactions,
        target_count=args.target_count,
        bins=args.bins,
        gmm_components=args.gmm_components,
        fourier_order=args.fourier_order,
        r2_threshold=args.r2_threshold,
        max_candidates=args.max_candidates,
    )
    print(f"    候选对数量: {len(valid_pairs):,}")
    print(f"    选中优质样本数量: {len(best_results)}")

    best_pair_df = pd.DataFrame([r.__dict__ for r in best_results])
    best_pair_df = best_pair_df.sort_values(["r2_gmm", "mse_gmm"], ascending=[False, True])

    print("[3/5] 生成可视化...")
    selected_data = df.merge(
        best_pair_df[["user_id", "category_id"]], on=["user_id", "category_id"], how="inner"
    )

    for r in best_results:
        pair_df = selected_data[
            (selected_data["user_id"] == r.user_id) & (selected_data["category_id"] == r.category_id)
        ]
        plot_pair(
            pair_df=pair_df,
            result=r,
            bins=args.bins,
            gmm_components=args.gmm_components,
            fourier_order=args.fourier_order,
            output_dir=args.output_dir,
        )

    print("[4/5] 保存结果表...")
    metrics_path = os.path.join(args.output_dir, "best5_multi_gaussian_vs_fourier.csv")
    best_pair_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("[5/5] 输出结论摘要...")
    gmm_better_ratio = float(np.mean(best_pair_df["gmm_better"].astype(int)))
    print(best_pair_df.to_string(index=False))
    print(f"\n多峰高斯在入选样本中的胜率: {gmm_better_ratio:.2%}")
    print(f"结果表: {metrics_path}")
    print(f"图像目录: {args.output_dir}")


if __name__ == "__main__":
    main()
