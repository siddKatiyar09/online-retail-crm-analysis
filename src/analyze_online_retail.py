from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


PALETTE = {
    "navy": "#19324d",
    "blue": "#2f6690",
    "teal": "#3ea3a8",
    "green": "#5aa469",
    "gold": "#d4a017",
    "orange": "#e07a5f",
    "red": "#c44536",
    "gray": "#6b7280",
    "light": "#f5f7fb",
    "border": "#d7deea",
    "ink": "#17202a",
}

CLUSTER_COLORS = ["#19324d", "#2f6690", "#3ea3a8", "#d4a017", "#e07a5f", "#5aa469"]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="CRM analysis for the Online Retail dataset.")
    parser.add_argument(
        "--input",
        default=str(project_root / "data" / "OnlineRetail.csv"),
        help="Path to the OnlineRetail CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "output"),
        help="Directory where reports, figures, and tables will be written.",
    )
    return parser.parse_args()


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_compact_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def safe_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return std


def load_transactions(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Place OnlineRetail.csv inside the data folder or pass --input."
        )
    raw = pd.read_csv(csv_path, encoding="ISO-8859-1")
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    raw["Revenue"] = raw["Quantity"] * raw["UnitPrice"]

    clean = raw.dropna(subset=["CustomerID"]).copy()
    clean = clean[(clean["Quantity"] > 0) & (clean["UnitPrice"] > 0)].copy()
    clean["CustomerID"] = clean["CustomerID"].astype(int)
    clean["Revenue"] = clean["Quantity"] * clean["UnitPrice"]
    clean["InvoiceMonth"] = clean["InvoiceDate"].dt.to_period("M").astype(str)
    clean["Description"] = clean["Description"].fillna("Unknown")
    return raw, clean


def summarize_quality(raw: pd.DataFrame, clean: pd.DataFrame) -> dict[str, float | int | str]:
    missing_customer_rows = int(raw["CustomerID"].isna().sum())
    cancelled_rows = int(raw["InvoiceNo"].astype(str).str.startswith("C").sum())
    negative_quantity_rows = int((raw["Quantity"] < 0).sum())
    zero_or_negative_price_rows = int((raw["UnitPrice"] <= 0).sum())
    uk_revenue_share = (
        clean.loc[clean["Country"] == "United Kingdom", "Revenue"].sum() / clean["Revenue"].sum() * 100
    )

    return {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "raw_customers": int(raw["CustomerID"].dropna().nunique()),
        "clean_customers": int(clean["CustomerID"].nunique()),
        "markets": int(clean["Country"].nunique()),
        "missing_customer_rows": missing_customer_rows,
        "missing_customer_pct": missing_customer_rows / len(raw) * 100,
        "cancelled_rows": cancelled_rows,
        "cancelled_pct": cancelled_rows / len(raw) * 100,
        "negative_quantity_rows": negative_quantity_rows,
        "negative_quantity_pct": negative_quantity_rows / len(raw) * 100,
        "zero_or_negative_price_rows": zero_or_negative_price_rows,
        "zero_or_negative_price_pct": zero_or_negative_price_rows / len(raw) * 100,
        "raw_revenue": float(raw["Revenue"].sum()),
        "clean_revenue": float(clean["Revenue"].sum()),
        "uk_revenue_share_pct": float(uk_revenue_share),
        "date_start": str(clean["InvoiceDate"].min()),
        "date_end": str(clean["InvoiceDate"].max()),
    }


def build_eda_tables(clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    monthly_revenue = (
        clean.groupby("InvoiceMonth")
        .agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"), Customers=("CustomerID", "nunique"))
        .reset_index()
    )

    top_countries = (
        clean.groupby("Country")
        .agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"), Customers=("CustomerID", "nunique"))
        .sort_values("Revenue", ascending=False)
        .head(10)
        .reset_index()
    )

    top_products = (
        clean.groupby("Description")
        .agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"), Orders=("InvoiceNo", "nunique"))
        .sort_values("Revenue", ascending=False)
        .head(10)
        .reset_index()
    )

    top_customers = (
        clean.groupby("CustomerID")
        .agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"), LastPurchase=("InvoiceDate", "max"))
        .sort_values("Revenue", ascending=False)
        .head(10)
        .reset_index()
    )

    return {
        "monthly_revenue": monthly_revenue,
        "top_countries": top_countries,
        "top_products": top_products,
        "top_customers": top_customers,
    }


def score_by_quantile(series: pd.Series, reverse: bool = False) -> pd.Series:
    ranked = series.rank(method="first", ascending=True)
    labels = [4, 3, 2, 1] if reverse else [1, 2, 3, 4]
    return pd.qcut(ranked, 4, labels=labels).astype(int)


def rfm_segment_label(row: pd.Series) -> str:
    r_score = int(row["R_Score"])
    f_score = int(row["F_Score"])
    m_score = int(row["M_Score"])

    if r_score >= 4 and f_score >= 4 and m_score >= 4:
        return "Champions"
    if f_score >= 4 and m_score >= 3 and r_score >= 2:
        return "Loyal Customers"
    if m_score >= 4 and r_score >= 2:
        return "Big Spenders"
    if r_score <= 2 and (f_score >= 3 or m_score >= 3):
        return "At Risk"
    if r_score >= 3 and f_score <= 2:
        return "Promising"
    return "Hibernating"


def build_customer_rfm(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_date = clean["InvoiceDate"].max() + pd.Timedelta(days=1)
    customer_summary = (
        clean.groupby("CustomerID")
        .agg(
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
            TotalItems=("Quantity", "sum"),
            UniqueProducts=("StockCode", "nunique"),
        )
        .reset_index()
    )

    customer_summary["Recency"] = (reference_date - customer_summary["LastPurchase"]).dt.days
    customer_summary["TenureDays"] = (customer_summary["LastPurchase"] - customer_summary["FirstPurchase"]).dt.days + 1
    customer_summary["AvgOrderValue"] = customer_summary["Monetary"] / customer_summary["Frequency"]
    customer_summary["ItemsPerOrder"] = customer_summary["TotalItems"] / customer_summary["Frequency"]
    customer_summary["R_Score"] = score_by_quantile(customer_summary["Recency"], reverse=True)
    customer_summary["F_Score"] = score_by_quantile(customer_summary["Frequency"])
    customer_summary["M_Score"] = score_by_quantile(customer_summary["Monetary"])
    customer_summary["RFM_Score"] = (
        customer_summary["R_Score"] + customer_summary["F_Score"] + customer_summary["M_Score"]
    )
    customer_summary["RFM_Segment"] = customer_summary.apply(rfm_segment_label, axis=1)

    segment_summary = (
        customer_summary.groupby("RFM_Segment")
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            Revenue=("Monetary", "sum"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    segment_summary["RevenueSharePct"] = segment_summary["Revenue"] / segment_summary["Revenue"].sum() * 100
    return customer_summary, segment_summary


def kmeans(matrix: np.ndarray, k: int, seed: int = 42, n_init: int = 12, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_labels = None
    best_centers = None
    best_inertia = None

    for _ in range(n_init):
        centers = matrix[rng.choice(len(matrix), size=k, replace=False)].copy()
        for _ in range(max_iter):
            distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            new_centers = np.vstack(
                [
                    matrix[labels == idx].mean(axis=0) if np.any(labels == idx) else centers[idx]
                    for idx in range(k)
                ]
            )
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers

        distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        inertia = float(distances[np.arange(len(matrix)), labels].sum())

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

    return best_labels, best_centers, best_inertia


def silhouette_sample(matrix: np.ndarray, labels: np.ndarray, sample_size: int = 1000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    if len(matrix) > sample_size:
        sample_idx = rng.choice(len(matrix), size=sample_size, replace=False)
        sample_matrix = matrix[sample_idx]
        sample_labels = labels[sample_idx]
    else:
        sample_matrix = matrix
        sample_labels = labels

    distances = np.sqrt(((sample_matrix[:, None, :] - sample_matrix[None, :, :]) ** 2).sum(axis=2))
    scores: list[float] = []

    for idx in range(len(sample_matrix)):
        same_cluster = sample_labels == sample_labels[idx]
        if same_cluster.sum() <= 1:
            scores.append(0.0)
            continue

        intra_distance = distances[idx, same_cluster].sum() / (same_cluster.sum() - 1)
        nearest_other_cluster = np.inf

        for cluster_id in np.unique(sample_labels):
            if cluster_id == sample_labels[idx]:
                continue
            other_cluster = sample_labels == cluster_id
            if other_cluster.sum() == 0:
                continue
            nearest_other_cluster = min(nearest_other_cluster, distances[idx, other_cluster].mean())

        scores.append((nearest_other_cluster - intra_distance) / max(intra_distance, nearest_other_cluster))

    return float(np.mean(scores))


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    metrics = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "RevenueSharePct"]
    standardized = profile[metrics].copy()
    for column in metrics:
        std = float(standardized[column].std(ddof=0))
        if std == 0:
            standardized[column] = 0.0
        else:
            standardized[column] = (standardized[column] - standardized[column].mean()) / std

    remaining = set(profile.index.tolist())
    names: dict[int, str] = {}

    champion = int(profile["Monetary"].idxmax())
    names[champion] = "Champions"
    remaining.remove(champion)

    if remaining:
        hibernating = int(
            (standardized.loc[list(remaining), "Recency"] - standardized.loc[list(remaining), "Frequency"] - standardized.loc[list(remaining), "Monetary"]).idxmax()
        )
        names[hibernating] = "Hibernating"
        remaining.remove(hibernating)

    if remaining:
        big_ticket = int(
            (
                standardized.loc[list(remaining), "AvgOrderValue"]
                + 0.6 * standardized.loc[list(remaining), "Monetary"]
                - 0.3 * standardized.loc[list(remaining), "Frequency"]
            ).idxmax()
        )
        names[big_ticket] = "Big-Ticket Occasionals"
        remaining.remove(big_ticket)

    for cluster_id in sorted(remaining):
        names[int(cluster_id)] = "Core Growth"

    return names


def build_customer_clusters(customer_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "TotalItems"]
    features = np.log1p(customer_summary[feature_columns].to_numpy(dtype=float))
    features = (features - features.mean(axis=0)) / safe_std(features)

    diagnostics_rows = []
    solutions: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}
    for k in range(2, 8):
        labels, centers, inertia = kmeans(features, k=k, seed=42 + k)
        silhouette = silhouette_sample(features, labels, sample_size=min(1000, len(features)), seed=42 + k)
        diagnostics_rows.append(
            {
                "K": k,
                "Inertia": round(inertia, 2),
                "SampledSilhouette": round(silhouette, 4),
            }
        )
        solutions[k] = (labels, centers, inertia, silhouette)

    diagnostics = pd.DataFrame(diagnostics_rows)

    selected_k = 4
    labels, _, _, _ = solutions[selected_k]
    clustered = customer_summary.copy()
    clustered["Cluster"] = labels.astype(int)

    profile = (
        clustered.groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
            AvgOrderValue=("AvgOrderValue", "mean"),
            TotalItems=("TotalItems", "mean"),
        )
        .reset_index()
    )
    profile["RevenueSharePct"] = clustered.groupby("Cluster")["Monetary"].sum().values / clustered["Monetary"].sum() * 100
    cluster_names = name_clusters(profile.set_index("Cluster"))
    profile["ClusterName"] = profile["Cluster"].map(cluster_names)
    clustered["ClusterName"] = clustered["Cluster"].map(cluster_names)
    return clustered, diagnostics, profile


def render_svg(title: str, subtitle: str, width: int, height: int, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
    .title {{ font: 700 22px Arial, sans-serif; fill: {PALETTE["ink"]}; }}
    .subtitle {{ font: 400 13px Arial, sans-serif; fill: {PALETTE["gray"]}; }}
    .axis {{ stroke: {PALETTE["gray"]}; stroke-width: 1; }}
    .grid {{ stroke: #e8edf5; stroke-width: 1; }}
    .label {{ font: 12px Arial, sans-serif; fill: {PALETTE["ink"]}; }}
    .value {{ font: 700 12px Arial, sans-serif; fill: {PALETTE["ink"]}; }}
</style>
<rect width="100%" height="100%" fill="white" rx="18" ry="18"/>
<text x="28" y="34" class="title">{html.escape(title)}</text>
<text x="28" y="56" class="subtitle">{html.escape(subtitle)}</text>
{body}
</svg>"""


def make_line_chart(labels: list[str], values: list[float], title: str, subtitle: str) -> str:
    width, height = 980, 460
    left, right, top, bottom = 80, 30, 90, 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(values) * 1.1
    min_value = 0.0

    points = []
    for idx, value in enumerate(values):
        x = left + idx * plot_width / max(1, len(values) - 1)
        y = top + plot_height - ((value - min_value) / max_value) * plot_height
        points.append((x, y))

    area_points = " ".join([f"{x:.1f},{y:.1f}" for x, y in points] + [f"{points[-1][0]:.1f},{top + plot_height}", f"{points[0][0]:.1f},{top + plot_height}"])
    line_points = " ".join([f"{x:.1f},{y:.1f}" for x, y in points])

    grid_lines = []
    for step in range(6):
        y = top + step * plot_height / 5
        value = max_value * (1 - step / 5)
        grid_lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid" />')
        grid_lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="label">{format_compact_number(value)}</text>')

    label_nodes = []
    for idx, label in enumerate(labels):
        x = left + idx * plot_width / max(1, len(labels) - 1)
        label_nodes.append(f'<text x="{x:.1f}" y="{height - 20}" text-anchor="middle" class="label">{html.escape(label)}</text>')

    point_nodes = []
    for (x, y), value in zip(points, values):
        point_nodes.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{PALETTE["blue"]}" />')
        point_nodes.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" class="value">{format_compact_number(value)}</text>')

    body = f"""
{''.join(grid_lines)}
<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" class="axis" />
<polygon points="{area_points}" fill="{PALETTE["teal"]}" opacity="0.18"/>
<polyline points="{line_points}" fill="none" stroke="{PALETTE["blue"]}" stroke-width="3"/>
{''.join(point_nodes)}
{''.join(label_nodes)}
"""
    return render_svg(title, subtitle, width, height, body)


def make_horizontal_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    subtitle: str,
    color: str,
    formatter,
) -> str:
    width = 980
    height = 110 + len(labels) * 42
    left, right, top = 260, 50, 90
    plot_width = width - left - right
    max_value = max(values) * 1.12

    body_parts = [f'<line x1="{left}" y1="{top + len(labels) * 42}" x2="{width - right}" y2="{top + len(labels) * 42}" class="axis" />']
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = top + idx * 42
        bar_width = (value / max_value) * plot_width
        body_parts.append(f'<text x="{left - 12}" y="{y + 20}" text-anchor="end" class="label">{html.escape(str(label))}</text>')
        body_parts.append(f'<rect x="{left}" y="{y + 8}" width="{bar_width:.1f}" height="20" rx="8" fill="{color}" opacity="0.88" />')
        body_parts.append(f'<text x="{left + bar_width + 10:.1f}" y="{y + 22}" class="value">{html.escape(formatter(value))}</text>')

    return render_svg(title, subtitle, width, height, "".join(body_parts))


def make_cluster_share_chart(profile: pd.DataFrame) -> str:
    labels = profile["ClusterName"].tolist()
    values = profile["RevenueSharePct"].tolist()
    width, height = 980, 420
    left, right, top, bottom = 90, 40, 90, 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    bar_width = plot_width / max(1, len(labels)) * 0.62
    max_value = max(values) * 1.15

    body_parts = []
    for step in range(5):
        y = top + step * plot_height / 4
        value = max_value * (1 - step / 4)
        body_parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid" />')
        body_parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="label">{value:.0f}%</text>')

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * plot_width / len(labels) + (plot_width / len(labels) - bar_width) / 2
        bar_height = (value / max_value) * plot_height
        y = top + plot_height - bar_height
        body_parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="12" fill="{CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]}" opacity="0.9" />')
        body_parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{value:.1f}%</text>')
        body_parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{height - 24}" text-anchor="middle" class="label">{html.escape(label)}</text>')

    body_parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" class="axis" />')
    return render_svg("Revenue Share by Cluster", "Four-cluster view chosen for CRM actionability", width, height, "".join(body_parts))


def make_scatter_chart(customer_summary: pd.DataFrame) -> str:
    sample = customer_summary.sample(min(1200, len(customer_summary)), random_state=42).copy()
    sample["RecencyScaled"] = np.log1p(sample["Recency"])
    sample["MonetaryScaled"] = np.log1p(sample["Monetary"])
    sample["SizeScaled"] = np.sqrt(sample["Frequency"]) * 2.2

    width, height = 980, 520
    left, right, top, bottom = 90, 40, 90, 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_min, x_max = sample["RecencyScaled"].min(), sample["RecencyScaled"].max()
    y_min, y_max = sample["MonetaryScaled"].min(), sample["MonetaryScaled"].max()

    body_parts = []
    for step in range(5):
        y = top + step * plot_height / 4
        value = np.expm1(y_max - step * (y_max - y_min) / 4)
        body_parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid" />')
        body_parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="label">{format_compact_number(value)}</text>')

    for step in range(5):
        x = left + step * plot_width / 4
        value = np.expm1(x_min + step * (x_max - x_min) / 4)
        body_parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" class="grid" />')
        body_parts.append(f'<text x="{x:.1f}" y="{height - 28}" text-anchor="middle" class="label">{value:.0f}d</text>')

    for row in sample.itertuples():
        x = left + (row.RecencyScaled - x_min) / max(1e-9, x_max - x_min) * plot_width
        y = top + plot_height - (row.MonetaryScaled - y_min) / max(1e-9, y_max - y_min) * plot_height
        color = CLUSTER_COLORS[int(row.Cluster) % len(CLUSTER_COLORS)]
        body_parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{max(2.6, min(10.0, row.SizeScaled)):.1f}" fill="{color}" opacity="0.42" />'
        )

    body_parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" class="axis" />')
    body_parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis" />')
    body_parts.append(f'<text x="{width / 2:.1f}" y="{height - 10}" text-anchor="middle" class="label">Recency in days (log scaled)</text>')
    body_parts.append(f'<text x="22" y="{height / 2:.1f}" transform="rotate(-90 22,{height / 2:.1f})" text-anchor="middle" class="label">Customer monetary value (log scaled)</text>')

    legend_y = top + 18
    for idx, cluster_name in enumerate(customer_summary[["Cluster", "ClusterName"]].drop_duplicates().sort_values("Cluster")["ClusterName"]):
        color = CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]
        x = width - 240
        y = legend_y + idx * 24
        body_parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" opacity="0.8" />')
        body_parts.append(f'<text x="{x + 14}" y="{y + 4}" class="label">{html.escape(cluster_name)}</text>')

    return render_svg(
        "Customer Value Map",
        "Sampled customers positioned by recency and monetary contribution",
        width,
        height,
        "".join(body_parts),
    )


def dataframe_to_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="data-table", border=0)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_html_report(
    output_dir: Path,
    summary: dict[str, float | int | str],
    eda_tables: dict[str, pd.DataFrame],
    segment_summary: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> str:
    monthly_peak = eda_tables["monthly_revenue"].sort_values("Revenue", ascending=False).iloc[0]
    top_country = eda_tables["top_countries"].iloc[0]
    champion_segment = segment_summary.sort_values("Revenue", ascending=False).iloc[0]
    champion_cluster = cluster_profile.sort_values("RevenueSharePct", ascending=False).iloc[0]

    cards = [
        ("Clean Revenue", format_currency(float(summary["clean_revenue"]))),
        ("Tracked Customers", f'{int(summary["clean_customers"]):,}'),
        ("Markets", f'{int(summary["markets"]):,}'),
        ("UK Revenue Share", format_percent(float(summary["uk_revenue_share_pct"]))),
    ]
    card_html = "".join(
        [
            f"""
            <div class="card">
                <div class="card-label">{html.escape(label)}</div>
                <div class="card-value">{html.escape(value)}</div>
            </div>
            """
            for label, value in cards
        ]
    )

    executive_points = [
        f"After CRM-focused cleaning, the dataset retains {summary['clean_rows']:,} valid transaction rows and {summary['clean_customers']:,} identified customers.",
        f"{float(summary['missing_customer_pct']):.1f}% of raw rows have no CustomerID, so anonymous orders should be treated separately from lifecycle CRM programs.",
        f"{top_country['Country']} contributes {format_currency(float(top_country['Revenue']))} and dominates geographic concentration.",
        f"The revenue peak lands in {monthly_peak['InvoiceMonth']} at {format_currency(float(monthly_peak['Revenue']))}, supporting a strong Q4 demand pattern.",
        f"The highest-value RFM segment is {champion_segment['RFM_Segment']} and the top ML cluster is {champion_cluster['ClusterName']}.",
    ]

    cluster_profile_for_report = cluster_profile.copy()
    cluster_profile_for_report["Monetary"] = cluster_profile_for_report["Monetary"].map(format_currency)
    cluster_profile_for_report["AvgOrderValue"] = cluster_profile_for_report["AvgOrderValue"].map(format_currency)
    cluster_profile_for_report["RevenueSharePct"] = cluster_profile_for_report["RevenueSharePct"].map(format_percent)

    segment_report = segment_summary.copy()
    segment_report["AvgRecency"] = segment_report["AvgRecency"].round(1)
    segment_report["AvgFrequency"] = segment_report["AvgFrequency"].round(1)
    segment_report["AvgMonetary"] = segment_report["AvgMonetary"].map(format_currency)
    segment_report["Revenue"] = segment_report["Revenue"].map(format_currency)
    segment_report["RevenueSharePct"] = segment_report["RevenueSharePct"].map(format_percent)

    diagnostics_report = diagnostics.copy()
    diagnostics_report["Inertia"] = diagnostics_report["Inertia"].round(2)
    diagnostics_report["SampledSilhouette"] = diagnostics_report["SampledSilhouette"].round(4)

    top_products_report = eda_tables["top_products"].copy()
    top_products_report["Revenue"] = top_products_report["Revenue"].map(format_currency)
    top_countries_report = eda_tables["top_countries"].copy()
    top_countries_report["Revenue"] = top_countries_report["Revenue"].map(format_currency)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Online Retail CRM Analysis</title>
    <style>
        :root {{
            --ink: {PALETTE["ink"]};
            --muted: {PALETTE["gray"]};
            --navy: {PALETTE["navy"]};
            --blue: {PALETTE["blue"]};
            --teal: {PALETTE["teal"]};
            --gold: {PALETTE["gold"]};
            --panel: white;
            --bg: {PALETTE["light"]};
            --border: {PALETTE["border"]};
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: var(--bg);
            color: var(--ink);
            line-height: 1.55;
        }}
        .wrap {{
            max-width: 1180px;
            margin: 0 auto;
            padding: 32px 24px 64px;
        }}
        .hero {{
            background: linear-gradient(135deg, var(--navy), var(--blue));
            color: white;
            border-radius: 24px;
            padding: 32px;
            margin-bottom: 24px;
        }}
        .hero h1 {{
            margin: 0 0 10px;
            font-size: 34px;
        }}
        .hero p {{
            margin: 0;
            max-width: 860px;
            color: rgba(255, 255, 255, 0.92);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card, .section {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 10px 26px rgba(25, 50, 77, 0.08);
        }}
        .card {{
            padding: 20px;
        }}
        .card-label {{
            color: var(--muted);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .card-value {{
            margin-top: 6px;
            font-size: 32px;
            font-weight: 700;
        }}
        .section {{
            padding: 24px;
            margin-bottom: 24px;
        }}
        h2 {{
            margin: 0 0 14px;
            font-size: 24px;
        }}
        ul {{
            margin-top: 8px;
        }}
        .figure {{
            margin: 20px 0 8px;
        }}
        .figure img {{
            width: 100%;
            border-radius: 18px;
            border: 1px solid var(--border);
            background: white;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .data-table thead th {{
            text-align: left;
            padding: 10px 12px;
            background: #eef3fb;
        }}
        .data-table td {{
            padding: 10px 12px;
            border-top: 1px solid var(--border);
        }}
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 900px) {{
            .grid, .two-col {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="wrap">
        <section class="hero">
            <h1>Online Retail CRM Analysis</h1>
            <p>
                This project converts raw transaction logs into a CRM-ready customer view using targeted data cleaning,
                exploratory analysis, RFM segmentation, and a four-cluster customer model for campaign planning.
            </p>
        </section>

        <section class="grid">
            {card_html}
        </section>

        <section class="section">
            <h2>Executive Summary</h2>
            <ul>
                {''.join(f'<li>{html.escape(point)}</li>' for point in executive_points)}
            </ul>
        </section>

        <section class="section">
            <h2>Data Quality and Scope</h2>
            <ul>
                <li>{summary['raw_rows']:,} raw transactions from {summary['date_start']} to {summary['date_end']}.</li>
                <li>{summary['missing_customer_rows']:,} rows ({format_percent(float(summary['missing_customer_pct']))}) have no CustomerID.</li>
                <li>{summary['cancelled_rows']:,} rows ({format_percent(float(summary['cancelled_pct']))}) are cancellations.</li>
                <li>{summary['negative_quantity_rows']:,} rows ({format_percent(float(summary['negative_quantity_pct']))}) have negative quantities.</li>
                <li>{summary['zero_or_negative_price_rows']:,} rows ({format_percent(float(summary['zero_or_negative_price_pct']))}) have zero or negative pricing.</li>
            </ul>
        </section>

        <section class="section">
            <h2>Sales and Market Trends</h2>
            <div class="figure"><img src="figures/monthly_revenue.svg" alt="Monthly revenue chart" /></div>
            <div class="figure"><img src="figures/top_countries_revenue.svg" alt="Top countries by revenue" /></div>
            <div class="two-col">
                <div>
                    <h3>Top Countries</h3>
                    {dataframe_to_html(top_countries_report)}
                </div>
                <div>
                    <h3>Top Products</h3>
                    {dataframe_to_html(top_products_report)}
                </div>
            </div>
        </section>

        <section class="section">
            <h2>CRM Segmentation</h2>
            <div class="figure"><img src="figures/rfm_segment_counts.svg" alt="RFM segment count chart" /></div>
            <p>
                The RFM layer highlights lifecycle segments that marketing teams can act on immediately. Champions and loyal customers
                should receive retention, referral, and upsell campaigns, while at-risk customers should receive win-back offers.
            </p>
            {dataframe_to_html(segment_report)}
        </section>

        <section class="section">
            <h2>Machine Learning Clusters</h2>
            <div class="figure"><img src="figures/cluster_revenue_share.svg" alt="Revenue share by cluster" /></div>
            <div class="figure"><img src="figures/customer_value_map.svg" alt="Customer value scatter plot" /></div>
            <p>
                A four-cluster solution was selected for actionability. The sampled silhouette score peaks at two clusters,
                but that split mainly separates elite customers from everyone else. Four clusters preserve enough granularity for campaign design.
            </p>
            <div class="two-col">
                <div>
                    <h3>Cluster Diagnostics</h3>
                    {dataframe_to_html(diagnostics_report)}
                </div>
                <div>
                    <h3>Cluster Profile</h3>
                    {dataframe_to_html(cluster_profile_for_report)}
                </div>
            </div>
        </section>

        <section class="section">
            <h2>Recommended CRM Actions</h2>
            <ul>
                <li>Protect Champions with early-access offers, premium service, and referral incentives.</li>
                <li>Move Core Growth customers toward loyalty by increasing repeat-order cadence and basket size.</li>
                <li>Target Big-Ticket Occasionals with event-based replenishment reminders and curated bundles.</li>
                <li>Launch win-back and onboarding automations for Hibernating and Promising customers.</li>
            </ul>
        </section>
    </div>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw, clean = load_transactions(input_path)
    summary = summarize_quality(raw, clean)
    eda_tables = build_eda_tables(clean)
    customer_summary, segment_summary = build_customer_rfm(clean)
    clustered_customers, diagnostics, cluster_profile = build_customer_clusters(customer_summary)
    diagnostics = diagnostics.sort_values("K").reset_index(drop=True)
    cluster_profile = cluster_profile.sort_values("RevenueSharePct", ascending=False).reset_index(drop=True)

    enriched_customers = clustered_customers.copy()

    monthly_revenue = eda_tables["monthly_revenue"]
    top_countries = eda_tables["top_countries"]

    write_text(
        figures_dir / "monthly_revenue.svg",
        make_line_chart(
            monthly_revenue["InvoiceMonth"].tolist(),
            monthly_revenue["Revenue"].tolist(),
            "Monthly Revenue Trend",
            "CRM-ready revenue after removing anonymous and invalid transactions",
        ),
    )
    write_text(
        figures_dir / "top_countries_revenue.svg",
        make_horizontal_bar_chart(
            top_countries["Country"].tolist(),
            top_countries["Revenue"].tolist(),
            "Top Countries by Revenue",
            "Revenue concentration is heavily dominated by the United Kingdom",
            PALETTE["blue"],
            formatter=format_currency,
        ),
    )
    write_text(
        figures_dir / "rfm_segment_counts.svg",
        make_horizontal_bar_chart(
            segment_summary["RFM_Segment"].tolist(),
            segment_summary["Customers"].tolist(),
            "RFM Segment Distribution",
            "Customer counts by lifecycle segment",
            PALETTE["green"],
            formatter=lambda value: f"{int(round(value)):,}",
        ),
    )
    write_text(figures_dir / "cluster_revenue_share.svg", make_cluster_share_chart(cluster_profile))
    write_text(figures_dir / "customer_value_map.svg", make_scatter_chart(enriched_customers))

    report_html = build_html_report(output_dir, summary, eda_tables, segment_summary, cluster_profile, diagnostics)
    write_text(output_dir / "analysis_report.html", report_html)

    executive_summary = f"""# Online Retail CRM Analysis

## Snapshot
- Clean revenue: {format_currency(float(summary["clean_revenue"]))}
- CRM-eligible customers: {summary["clean_customers"]:,}
- Markets covered: {summary["markets"]:,}
- UK revenue concentration: {format_percent(float(summary["uk_revenue_share_pct"]))}

## Key Findings
- {format_percent(float(summary["missing_customer_pct"]))} of raw rows have no `CustomerID`, so anonymous orders need separate handling from retention campaigns.
- Peak cleaned revenue occurs in {monthly_revenue.sort_values("Revenue", ascending=False).iloc[0]["InvoiceMonth"]}.
- The United Kingdom contributes the majority of revenue, creating meaningful geographic concentration risk.
- RFM and clustering both show a small high-value customer tier generating a disproportionate share of revenue.

## CRM Actions
- Retain Champions with VIP treatment, referrals, and exclusive launches.
- Grow Core Growth customers with cross-sell bundles and reorder reminders.
- Win back Hibernating customers with limited-time incentives and reactivation flows.
- Treat Big-Ticket Occasionals with event-driven premium offers instead of high-frequency promotions.
"""
    write_text(output_dir / "executive_summary.md", executive_summary)

    export_customers = enriched_customers.copy()
    export_customers["FirstPurchase"] = export_customers["FirstPurchase"].astype(str)
    export_customers["LastPurchase"] = export_customers["LastPurchase"].astype(str)
    export_customers.to_csv(output_dir / "customer_rfm_segments.csv", index=False)
    segment_summary.to_csv(output_dir / "rfm_segment_summary.csv", index=False)
    cluster_profile.to_csv(output_dir / "cluster_profile.csv", index=False)
    diagnostics.to_csv(output_dir / "kmeans_diagnostics.csv", index=False)
    eda_tables["monthly_revenue"].to_csv(output_dir / "monthly_revenue.csv", index=False)
    eda_tables["top_products"].to_csv(output_dir / "top_products.csv", index=False)

    summary_json = {
        "dataset": input_path.name,
        "clean_revenue": round(float(summary["clean_revenue"]), 2),
        "clean_customers": int(summary["clean_customers"]),
        "markets": int(summary["markets"]),
        "uk_revenue_share_pct": round(float(summary["uk_revenue_share_pct"]), 2),
        "top_rfm_segment": str(segment_summary.sort_values("Revenue", ascending=False).iloc[0]["RFM_Segment"]),
        "top_cluster": str(cluster_profile.sort_values("RevenueSharePct", ascending=False).iloc[0]["ClusterName"]),
    }
    write_text(output_dir / "analysis_summary.json", json.dumps(summary_json, indent=2))

    print(f"Analysis written to: {output_dir}")


if __name__ == "__main__":
    main()
