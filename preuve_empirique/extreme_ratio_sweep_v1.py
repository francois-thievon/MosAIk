import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import multivariate_normal, spearmanr
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader, TensorDataset


ResultValue = Union[float, int]
ResultRow = dict[str, ResultValue]


@dataclass
class ExperimentConfig:
    output_dir: Path
    n_samples: int = 5_000
    n_blobs: int = 2
    rank: int = 2
    repeats: int = 50
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_dim: int = 48
    kde_bandwidth: float = 0.5
    threshold: float = 0.2
    total_ratios: int = 30
    low_ratio_points: int = 8
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-3
    weight_decay: float = 1e-5
    base_seed: int = 1234
    show_plot: bool = True
    save_raw_results: bool = True


class ProbabilisticAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, rank: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, input_dim)
        self.v_head = nn.Linear(hidden_dim, input_dim * rank)
        self.d_head = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.decoder_hidden(z)
        mu = self.mu_head(h)
        v = self.v_head(h).view(-1, self.input_dim, self.rank)
        d_diag = torch.exp(self.d_head(h)) + 1e-4
        return mu, v, d_diag

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        mu, v, d_diag = self.decode(z)
        return z, mu, v, d_diag


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probabilistic_nll_loss(
    x: torch.Tensor,
    mu: torch.Tensor,
    v: torch.Tensor,
    d_diag: torch.Tensor,
) -> torch.Tensor:
    v_vt = torch.bmm(v, v.transpose(1, 2))
    sigma = v_vt + torch.diag_embed(d_diag)
    dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
    return -dist.log_prob(x).mean()


def generate_data(
    n_samples: int,
    n_features: int,
    n_blobs: int,
    rank: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v_matrix = rng.standard_normal((n_features, rank))
    true_cov = v_matrix @ v_matrix.T + 0.1 * np.eye(n_features)
    means = [rng.uniform(-5.0, 5.0, n_features) for _ in range(n_blobs)]
    distributions = [multivariate_normal(mean=mean, cov=true_cov) for mean in means]

    counts = [n_samples // n_blobs] * n_blobs
    for idx in range(n_samples % n_blobs):
        counts[idx] += 1

    batches = []
    for mean, count in zip(means, counts):
        samples = rng.multivariate_normal(mean, true_cov, count)
        batches.append(samples)

    x = np.vstack(batches).astype(np.float32)
    probs = np.array([dist.pdf(x) for dist in distributions])
    px_combined = np.mean(probs, axis=0)
    log_px = np.log(px_combined + 1e-10)
    return x, log_px, true_cov


def build_ratio_pairs(
    candidate_dims: list[int],
    candidate_bottlenecks: list[int],
    threshold: float,
    total_ratios: int,
    low_ratio_points: int,
) -> list[tuple[int, int, float]]:
    ratio_to_pair: dict[float, tuple[int, int]] = {}
    for dim in candidate_dims:
        for bottleneck in candidate_bottlenecks:
            if bottleneck <= dim:
                ratio = round(bottleneck / dim, 4)
                current = ratio_to_pair.get(ratio)
                if current is None or dim > current[0]:
                    ratio_to_pair[ratio] = (dim, bottleneck)

    available = np.array(sorted(ratio_to_pair.keys()), dtype=float)
    low_pool = available[available <= threshold]
    high_pool = available[available > threshold]
    if len(high_pool) == 0:
        raise ValueError("Aucun ratio > threshold n'est disponible.")

    high_ratio_points = max(total_ratios - low_ratio_points, 1)
    low_ratio_points = min(low_ratio_points, len(low_pool))
    high_ratio_points = min(high_ratio_points, len(high_pool))

    if low_ratio_points > 0 and len(low_pool) > 0:
        low_targets = np.linspace(low_pool.min(), threshold, low_ratio_points)
    else:
        low_targets = np.array([], dtype=float)
    high_targets = np.linspace(max(threshold, high_pool.min()), high_pool.max(), high_ratio_points)

    def pick_nearest_unique(targets: np.ndarray, pool: np.ndarray, used: set[float]) -> list[float]:
        chosen: list[float] = []
        if len(pool) == 0:
            return chosen
        for target in targets:
            order = np.argsort(np.abs(pool - target))
            selected = None
            for idx in order:
                candidate = float(pool[idx])
                if candidate not in used:
                    selected = candidate
                    break
            if selected is None:
                selected = float(pool[order[0]])
            used.add(selected)
            chosen.append(selected)
        return chosen

    used: set[float] = set()
    selected = pick_nearest_unique(low_targets, low_pool, used)
    selected += pick_nearest_unique(high_targets, high_pool, used)

    if len(selected) < total_ratios:
        remaining = [float(value) for value in available if float(value) not in used]
        order = sorted(remaining, key=lambda value: (value <= threshold, abs(value - threshold), value), reverse=True)
        selected.extend(order[: total_ratios - len(selected)])

    selected = sorted(selected)
    return [(ratio_to_pair[ratio][0], ratio_to_pair[ratio][1], ratio) for ratio in selected]


def train_model(
    model: ProbabilisticAE,
    x: np.ndarray,
    device: torch.device,
    config: ExperimentConfig,
) -> float:
    x_tensor = torch.from_numpy(x).float()
    dataset = DataLoader(TensorDataset(x_tensor), batch_size=config.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_loss = math.inf
    patience = 0
    final_loss = math.inf

    model.train()
    for _ in range(config.epochs):
        epoch_loss = 0.0
        for batch in dataset:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            _, mu, v, d_diag = model(x_batch)
            loss = probabilistic_nll_loss(x_batch, mu, v, d_diag)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        final_loss = epoch_loss / max(len(dataset), 1)
        if final_loss < best_loss - config.early_stopping_min_delta:
            best_loss = final_loss
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    return final_loss


def evaluate_spearman(
    model: ProbabilisticAE,
    x: np.ndarray,
    log_px: np.ndarray,
    device: torch.device,
    bandwidth: float,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        z = model.encode(torch.from_numpy(x).float().to(device)).cpu().numpy()

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(z)
    log_pz = kde.score_samples(z)
    corr, pvalue = spearmanr(log_px, log_pz)
    return float(corr), float(pvalue)


def fit_semilog_line(ratios: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    xlog = np.log10(ratios)
    coefficients = np.polyfit(xlog, values, 1)
    fitted = np.polyval(coefficients, xlog)
    ss_res = float(np.sum((values - fitted) ** 2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coefficients, fitted, r2


def run_repeat(
    repeat_idx: int,
    selected_pairs: list[tuple[int, int, float]],
    config: ExperimentConfig,
    device: torch.device,
) -> list[ResultRow]:
    results: list[ResultRow] = []
    repeat_seed = config.base_seed + repeat_idx

    for pair_idx, (dim, bottleneck, ratio) in enumerate(selected_pairs):
        seed = repeat_seed * 10_000 + pair_idx
        set_seed(seed)
        rng = np.random.default_rng(seed)
        x, log_px, _ = generate_data(
            n_samples=config.n_samples,
            n_features=dim,
            n_blobs=config.n_blobs,
            rank=config.rank,
            rng=rng,
        )

        model = ProbabilisticAE(
            input_dim=dim,
            latent_dim=bottleneck,
            rank=config.rank,
            hidden_dim=config.hidden_dim,
        ).to(device)

        final_loss = train_model(model, x, device, config)
        spearman_value, pvalue = evaluate_spearman(
            model=model,
            x=x,
            log_px=log_px,
            device=device,
            bandwidth=config.kde_bandwidth,
        )
        results.append(
            {
                "repeat": repeat_idx,
                "dim": dim,
                "bottleneck": bottleneck,
                "ratio": ratio,
                "spearman": spearman_value,
                "pvalue": pvalue,
                "final_loss": final_loss,
            }
        )

    return results


def aggregate_results(raw_results: list[ResultRow]) -> list[dict[str, float]]:
    grouped: dict[float, list[ResultRow]] = {}
    for row in raw_results:
        ratio = float(row["ratio"])
        grouped.setdefault(ratio, []).append(row)

    aggregated: list[dict[str, float]] = []
    for ratio in sorted(grouped):
        rows = grouped[ratio]
        spearman_values = np.array([float(row["spearman"]) for row in rows], dtype=float)
        losses = np.array([float(row["final_loss"]) for row in rows], dtype=float)
        dims = sorted({int(row["dim"]) for row in rows})
        bottlenecks = sorted({int(row["bottleneck"]) for row in rows})
        std = float(np.std(spearman_values, ddof=1)) if len(spearman_values) > 1 else 0.0
        sem = std / math.sqrt(len(spearman_values)) if len(spearman_values) > 0 else 0.0
        aggregated.append(
            {
                "ratio": ratio,
                "spearman_mean": float(np.mean(spearman_values)),
                "spearman_std": std,
                "spearman_sem": sem,
                "spearman_min": float(np.min(spearman_values)),
                "spearman_max": float(np.max(spearman_values)),
                "loss_mean": float(np.mean(losses)),
                "loss_std": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
                "count": float(len(spearman_values)),
                "dim": float(max(dims)),
                "bottleneck": float(max(bottlenecks)),
            }
        )
    return aggregated


def summarize_repeat_linearity(raw_results: list[ResultRow]) -> dict[str, float]:
    repeat_to_rows: dict[int, list[ResultRow]] = {}
    for row in raw_results:
        repeat_to_rows.setdefault(int(row["repeat"]), []).append(row)

    r2_values = []
    slopes = []
    for repeat_idx in sorted(repeat_to_rows):
        rows = sorted(repeat_to_rows[repeat_idx], key=lambda item: float(item["ratio"]))
        ratios = np.array([float(row["ratio"]) for row in rows], dtype=float)
        values = np.array([float(row["spearman"]) for row in rows], dtype=float)
        coefficients, _, r2 = fit_semilog_line(ratios, values)
        slopes.append(float(coefficients[0]))
        r2_values.append(float(r2))

    r2_array = np.array(r2_values, dtype=float)
    slope_array = np.array(slopes, dtype=float)
    return {
        "r2_mean": float(np.mean(r2_array)),
        "r2_std": float(np.std(r2_array, ddof=1)) if len(r2_array) > 1 else 0.0,
        "r2_q10": float(np.quantile(r2_array, 0.10)),
        "r2_q50": float(np.quantile(r2_array, 0.50)),
        "r2_q90": float(np.quantile(r2_array, 0.90)),
        "slope_mean": float(np.mean(slope_array)),
        "slope_std": float(np.std(slope_array, ddof=1)) if len(slope_array) > 1 else 0.0,
    }


def save_csv(path: Path, rows: list[ResultRow]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    raw_results: list[ResultRow],
    aggregated: list[dict[str, float]],
    linearity_summary: dict[str, float],
    output_path: Path,
    config: ExperimentConfig,
) -> None:
    ratios = np.array([row["ratio"] for row in aggregated], dtype=float)
    mean_values = np.array([row["spearman_mean"] for row in aggregated], dtype=float)
    std_values = np.array([row["spearman_std"] for row in aggregated], dtype=float)
    sem_values = np.array([row["spearman_sem"] for row in aggregated], dtype=float)
    coefficients, fitted, r2_mean_curve = fit_semilog_line(ratios, mean_values)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    repeat_to_rows: dict[int, list[ResultRow]] = {}
    for row in raw_results:
        repeat_to_rows.setdefault(int(row["repeat"]), []).append(row)

    for repeat_idx, rows in repeat_to_rows.items():
        rows_sorted = sorted(rows, key=lambda item: float(item["ratio"]))
        x = np.array([float(row["ratio"]) for row in rows_sorted], dtype=float)
        y = np.array([float(row["spearman"]) for row in rows_sorted], dtype=float)
        axes[0, 0].plot(x, y, color="0.75", alpha=0.18, linewidth=1)
        axes[0, 1].semilogx(x, y, color="0.75", alpha=0.18, linewidth=1)

    axes[0, 0].plot(ratios, mean_values, color="#c2410c", marker="o", linewidth=2.2, label="Moyenne")
    axes[0, 0].fill_between(
        ratios,
        np.clip(mean_values - std_values, 0.0, 1.0),
        np.clip(mean_values + std_values, 0.0, 1.0),
        color="#fb923c",
        alpha=0.25,
        label="Moyenne ± 1 écart-type",
    )
    axes[0, 0].set_title("Spearman moyen vs ratio")
    axes[0, 0].set_xlabel("Ratio bottleneck / dimension")
    axes[0, 0].set_ylabel("Spearman")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)
    axes[0, 0].legend()

    axes[0, 1].semilogx(ratios, mean_values, color="#0f766e", marker="o", linewidth=2.2, label="Moyenne")
    axes[0, 1].semilogx(
        ratios,
        fitted,
        linestyle="--",
        color="#115e59",
        linewidth=2.0,
        label=(
            f"Fit moyen: y = {coefficients[0]:.3f} log10(x) + {coefficients[1]:.3f}\n"
            f"R² courbe moyenne = {r2_mean_curve:.3f}"
        ),
    )
    axes[0, 1].fill_between(
        ratios,
        np.clip(mean_values - sem_values, 0.0, 1.0),
        np.clip(mean_values + sem_values, 0.0, 1.0),
        color="#14b8a6",
        alpha=0.22,
        label="Moyenne ± SEM",
    )
    axes[0, 1].set_title("Vue semi-log pour tester la quasi-linéarité")
    axes[0, 1].set_xlabel("Ratio bottleneck / dimension (log)")
    axes[0, 1].set_ylabel("Spearman")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)
    axes[0, 1].legend()

    axes[1, 0].plot(ratios, std_values, color="#7c3aed", marker="o", linewidth=2)
    axes[1, 0].set_title("Variance du Spearman selon le ratio")
    axes[1, 0].set_xlabel("Ratio bottleneck / dimension")
    axes[1, 0].set_ylabel("Écart-type sur 50 répétitions")
    axes[1, 0].grid(True, linestyle="--", alpha=0.4)

    r2_values = []
    for repeat_idx in sorted(repeat_to_rows):
        rows = sorted(repeat_to_rows[repeat_idx], key=lambda item: float(item["ratio"]))
        repeat_ratios = np.array([float(row["ratio"]) for row in rows], dtype=float)
        repeat_spearman = np.array([float(row["spearman"]) for row in rows], dtype=float)
        _, _, r2 = fit_semilog_line(repeat_ratios, repeat_spearman)
        r2_values.append(r2)

    axes[1, 1].hist(r2_values, bins=12, color="#2563eb", alpha=0.8, edgecolor="white")
    axes[1, 1].axvline(linearity_summary["r2_mean"], color="#1d4ed8", linestyle="--", linewidth=2)
    axes[1, 1].set_title("Distribution du R² semi-log sur les répétitions")
    axes[1, 1].set_xlabel("R²")
    axes[1, 1].set_ylabel("Fréquence")
    axes[1, 1].grid(True, linestyle="--", alpha=0.25)
    axes[1, 1].text(
        0.05,
        0.95,
        (
            f"R² moyen = {linearity_summary['r2_mean']:.3f} ± {linearity_summary['r2_std']:.3f}\n"
            f"Q10 = {linearity_summary['r2_q10']:.3f}\n"
            f"Médiane = {linearity_summary['r2_q50']:.3f}\n"
            f"Q90 = {linearity_summary['r2_q90']:.3f}\n"
            f"Pente moyenne = {linearity_summary['slope_mean']:.3f} ± {linearity_summary['slope_std']:.3f}"
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.suptitle(
        (
            f"Sweep extrême Spearman/ratio | n={config.n_samples} | repeats={config.repeats} | "
            f"ratios={len(ratios)}"
        ),
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    if config.show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Sweep extrême du ratio bottleneck/dimension avec répétitions et analyse de variance."
    )
    default_output_dir = Path(__file__).resolve().parent / "extreme_ratio_sweep_v1_outputs"
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--kde-bandwidth", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--total-ratios", type=int, default=30)
    parser.add_argument("--low-ratio-points", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    return ExperimentConfig(
        output_dir=args.output_dir,
        n_samples=args.samples,
        repeats=args.repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        kde_bandwidth=args.kde_bandwidth,
        threshold=args.threshold,
        total_ratios=args.total_ratios,
        low_ratio_points=args.low_ratio_points,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        weight_decay=args.weight_decay,
        base_seed=args.seed,
        show_plot=not args.no_show,
    )


def main() -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidate_dims = [16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 200]
    candidate_bottlenecks = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
    selected_pairs = build_ratio_pairs(
        candidate_dims=candidate_dims,
        candidate_bottlenecks=candidate_bottlenecks,
        threshold=config.threshold,
        total_ratios=config.total_ratios,
        low_ratio_points=config.low_ratio_points,
    )

    print(f"Device: {device}")
    print(f"Nombre de ratios tests: {len(selected_pairs)}")
    print(f"Ratios selectionnes: {[round(pair[2], 3) for pair in selected_pairs]}")
    print(
        "Attention: cette v1 est volontairement extreme. "
        "Avec 10 000 points, 50 repetitions et plusieurs dizaines de ratios, le runtime peut etre tres long."
    )

    raw_results: list[ResultRow] = []
    total_runs = config.repeats * len(selected_pairs)
    completed_runs = 0

    for repeat_idx in range(config.repeats):
        repeat_results = run_repeat(repeat_idx, selected_pairs, config, device)
        raw_results.extend(repeat_results)
        completed_runs += len(repeat_results)

        current_mean = np.mean([float(row["spearman"]) for row in repeat_results])
        print(
            f"Repeat {repeat_idx + 1:>2}/{config.repeats} termine | "
            f"Spearman moyen du repeat = {current_mean:.4f} | "
            f"Progression = {completed_runs}/{total_runs}"
        )

        if config.save_raw_results:
            save_csv(config.output_dir / "raw_results_partial.csv", raw_results)

    aggregated = aggregate_results(raw_results)
    linearity_summary = summarize_repeat_linearity(raw_results)

    if config.save_raw_results:
        save_csv(config.output_dir / "raw_results.csv", raw_results)
    save_csv(config.output_dir / "aggregated_results.csv", aggregated)

    plot_results(
        raw_results=raw_results,
        aggregated=aggregated,
        linearity_summary=linearity_summary,
        output_path=config.output_dir / "extreme_ratio_sweep.png",
        config=config,
    )

    print("Resume linearite semi-log:")
    print(
        f"R2 moyen = {linearity_summary['r2_mean']:.4f} +- {linearity_summary['r2_std']:.4f} | "
        f"Q10 = {linearity_summary['r2_q10']:.4f} | Mediane = {linearity_summary['r2_q50']:.4f} | "
        f"Q90 = {linearity_summary['r2_q90']:.4f}"
    )
    print(
        f"Pente moyenne = {linearity_summary['slope_mean']:.4f} +- {linearity_summary['slope_std']:.4f}"
    )
    print(f"Sorties ecrites dans: {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()