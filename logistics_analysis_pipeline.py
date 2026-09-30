"""Reproducible synthetic last-mile logistics analysis for the internship reports.

Requirements: Python 3.10+, pandas, NumPy, Pillow.

This script creates an explicitly synthetic shipment dataset, applies the documented
quality rules, calculates descriptive and predictive metrics, and saves chart PNGs.
It contains no real customer, courier, or company data.
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent / "work"
CHART_DIR = ROOT / "charts"
SEED = 20260930
ZONES = ["Central", "North", "South", "East", "West", "Outer"]
TRAFFIC = ["Low", "Medium", "High"]
WEATHER = ["Clear", "Rain"]
NUMERIC_FEATURES = ["distance_km", "package_weight_kg", "dispatch_hour", "day_of_week"]
CATEGORICAL_FEATURES = ["zone", "traffic_level", "weather"]


def _choose(rng: np.random.Generator, candidates: np.ndarray, count: int) -> np.ndarray:
    return rng.choice(candidates, size=count, replace=False)


def build_raw_data(seed: int = SEED) -> pd.DataFrame:
    """Create 1,200 synthetic shipments plus 15 deliberate duplicate rows and defects."""
    rng = np.random.default_rng(seed)
    n = 1200
    dates = pd.Timestamp("2026-06-01") + pd.to_timedelta(rng.integers(0, 90, n), unit="D")
    zone = rng.choice(ZONES, size=n, p=[0.19, 0.17, 0.17, 0.16, 0.16, 0.15])
    distance = np.clip(rng.gamma(shape=2.35, scale=3.15, size=n) + 0.7, 0.7, 22.0)
    traffic = rng.choice(TRAFFIC, size=n, p=[0.27, 0.45, 0.28]).astype(object)
    weather = rng.choice(WEATHER, size=n, p=[0.81, 0.19]).astype(object)
    weight = np.clip(rng.gamma(shape=2.0, scale=1.15, size=n) + 0.1, 0.1, 18.0)
    dispatch_hour = rng.integers(8, 22, n)
    day_of_week = pd.DatetimeIndex(dates).dayofweek.to_numpy()

    traffic_delay = pd.Series(traffic).map({"Low": 0.0, "Medium": 10.0, "High": 24.0}).to_numpy()
    weather_delay = pd.Series(weather).map({"Clear": 0.0, "Rain": 10.0}).to_numpy()
    peak_delay = np.where((dispatch_hour >= 17) & (dispatch_hour <= 20), 13.0,
                          np.where((dispatch_hour >= 8) & (dispatch_hour <= 10), 7.0, 0.0))
    zone_delay = pd.Series(zone).map({"Central": 4.0, "North": 1.0, "South": 3.0,
                                      "East": 2.0, "West": 0.0, "Outer": 8.0}).to_numpy()
    weekend_delay = np.where(day_of_week >= 5, 3.0, 0.0)
    high_traffic = (traffic == "High").astype(float)
    actual = (22.0 + 4.1 * distance + traffic_delay + weather_delay + peak_delay
              + 0.7 * weight + zone_delay + 0.45 * distance * high_traffic
              + weekend_delay + rng.normal(0, 9.0, n))
    actual = np.maximum(14.0, actual)
    promised = np.maximum(42.0, 50.0 + 4.8 * distance + rng.normal(0, 4.0, n))
    cost = np.maximum(18.0, 30.0 + 4.8 * distance + 1.7 * weight
                      + 5.0 * (weather == "Rain") + rng.normal(0, 5.0, n))
    attempts = np.where(actual - promised > 55, 2, 1).astype(int)
    retries = rng.random(n) < 0.035
    attempts = np.where(retries, np.minimum(attempts + 1, 3), attempts)

    base = pd.DataFrame({
        "shipment_id": [f"PU{idx:05d}" for idx in range(1, n + 1)],
        "order_date": pd.to_datetime(dates),
        "zone": zone,
        "distance_km": distance.round(2),
        "traffic_level": traffic,
        "weather": weather,
        "package_weight_kg": weight.round(2),
        "dispatch_hour": dispatch_hour,
        "planned_delivery_min": promised.round(1),
        "actual_delivery_min": actual.round(1),
        "delivery_cost_inr": cost.round(2),
        "attempt_count": attempts,
    })
    base["on_time"] = (base["actual_delivery_min"] <= base["planned_delivery_min"])

    # Deliberate quality problems: exact missingness counts, invalid ranges, label errors.
    all_idx = np.arange(n)
    missing_distance = _choose(rng, all_idx, 36)
    missing_weight = _choose(rng, np.setdiff1d(all_idx, missing_distance), 30)
    missing_weather = _choose(rng, np.setdiff1d(all_idx, np.r_[missing_distance, missing_weight]), 24)
    missing_target = _choose(rng, np.setdiff1d(all_idx, np.r_[missing_distance, missing_weight, missing_weather]), 12)
    base.loc[missing_distance, "distance_km"] = np.nan
    base.loc[missing_weight, "package_weight_kg"] = np.nan
    base.loc[missing_weather, "weather"] = None
    base.loc[missing_target, "actual_delivery_min"] = np.nan

    valid_distance_idx = np.setdiff1d(all_idx, missing_distance)
    negative_distance = _choose(rng, valid_distance_idx, 5)
    remaining = np.setdiff1d(valid_distance_idx, negative_distance)
    far_distance = _choose(rng, remaining, 5)
    base.loc[negative_distance, "distance_km"] = -3.0
    base.loc[far_distance, "distance_km"] = 120.0

    valid_target_idx = np.setdiff1d(all_idx, missing_target)
    bad_target = _choose(rng, valid_target_idx, 3)
    base.loc[bad_target, "actual_delivery_min"] = 999.0
    base.loc[bad_target, "on_time"] = False
    messy_traffic = _choose(rng, all_idx, 8)
    base.loc[messy_traffic, "traffic_level"] = base.loc[messy_traffic, "traffic_level"].map(
        lambda x: f" {str(x).lower()} "
    )

    duplicates = base.iloc[_choose(rng, all_idx, 15)].copy()
    raw = pd.concat([base, duplicates], ignore_index=True)
    return raw


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply key, range, missing-value, and target-validity rules."""
    counts = {
        "raw_rows": int(len(raw)),
        "duplicate_rows": int(raw["shipment_id"].duplicated(keep="first").sum()),
        "missing_distance": int(raw["distance_km"].isna().sum()),
        "invalid_distance": int(((raw["distance_km"] <= 0) | (raw["distance_km"] > 50)).sum()),
        "missing_weight": int(raw["package_weight_kg"].isna().sum()),
        "missing_weather": int(raw["weather"].isna().sum()),
        "missing_target": int(raw["actual_delivery_min"].isna().sum()),
        "invalid_target": int((raw["actual_delivery_min"] > 360).sum()),
    }
    clean = raw.drop_duplicates(subset="shipment_id", keep="first").copy()
    counts["unique_missing_distance"] = int(clean["distance_km"].isna().sum())
    counts["unique_invalid_distance"] = int(((clean["distance_km"] <= 0) | (clean["distance_km"] > 50)).sum())
    counts["unique_missing_weight"] = int(clean["package_weight_kg"].isna().sum())
    counts["unique_missing_weather"] = int(clean["weather"].isna().sum())
    counts["unique_missing_target"] = int(clean["actual_delivery_min"].isna().sum())
    counts["unique_invalid_target"] = int((clean["actual_delivery_min"] > 360).sum())
    clean["order_date"] = pd.to_datetime(clean["order_date"], errors="coerce")
    for col in ["zone", "traffic_level", "weather"]:
        clean[col] = clean[col].astype("string").str.strip().str.title()
    clean["distance_km"] = pd.to_numeric(clean["distance_km"], errors="coerce")
    clean.loc[(clean["distance_km"] <= 0) | (clean["distance_km"] > 50), "distance_km"] = np.nan
    clean["package_weight_kg"] = pd.to_numeric(clean["package_weight_kg"], errors="coerce")
    clean["actual_delivery_min"] = pd.to_numeric(clean["actual_delivery_min"], errors="coerce")
    valid_target = clean["actual_delivery_min"].between(0, 360, inclusive="both")
    clean = clean.loc[valid_target].copy()
    clean["distance_km"] = clean.groupby("zone", dropna=False)["distance_km"].transform(
        lambda values: values.fillna(values.median())
    )
    clean["package_weight_kg"] = clean["package_weight_kg"].fillna(clean["package_weight_kg"].median())
    clean["weather"] = clean["weather"].fillna(clean["weather"].mode().iloc[0])
    clean["day_of_week"] = clean["order_date"].dt.dayofweek.astype(int)
    clean["on_time"] = clean["actual_delivery_min"] <= clean["planned_delivery_min"]
    clean["delivery_delay_min"] = np.maximum(
        0.0, clean["actual_delivery_min"] - clean["planned_delivery_min"]
    )
    counts["deduplicated_rows"] = int(len(raw) - counts["duplicate_rows"])
    counts["clean_rows"] = int(len(clean))
    return clean.sort_values(["order_date", "shipment_id"]).reset_index(drop=True), counts


def _feature_matrix(frame: pd.DataFrame, means: pd.Series | None = None,
                    stds: pd.Series | None = None) -> tuple[np.ndarray, list[str], pd.Series, pd.Series]:
    numeric = frame[NUMERIC_FEATURES].astype(float)
    if means is None:
        means = numeric.mean()
    if stds is None:
        stds = numeric.std(ddof=0).replace(0, 1.0)
    scaled = (numeric - means) / stds
    names = list(NUMERIC_FEATURES)
    parts = [scaled.to_numpy(dtype=float)]
    category_values = {"zone": ZONES, "traffic_level": TRAFFIC, "weather": WEATHER}
    for col in CATEGORICAL_FEATURES:
        values = category_values[col]
        for level in values[1:]:
            parts.append((frame[col].astype(str).to_numpy() == level).astype(float).reshape(-1, 1))
            names.append(f"{col}_{level}")
    matrix = np.column_stack([np.ones(len(frame)), *parts])
    return matrix, ["intercept", *names], means, stds


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - np.sum(errors ** 2) / denominator) if denominator else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def analyze(clean: pd.DataFrame) -> dict:
    """Calculate reproducible profile, EDA, time split, linear model and baseline."""
    y = clean["actual_delivery_min"].to_numpy(dtype=float)
    dates = sorted(clean["order_date"].dropna().unique())
    split_date = pd.Timestamp(dates[int(0.80 * len(dates))])
    train = clean.loc[clean["order_date"] < split_date].copy()
    test = clean.loc[clean["order_date"] >= split_date].copy()
    x_train, feature_names, means, stds = _feature_matrix(train)
    x_test, _, _, _ = _feature_matrix(test, means, stds)
    y_train = train["actual_delivery_min"].to_numpy(dtype=float)
    y_test = test["actual_delivery_min"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    predictions = x_test @ beta
    baseline_predictions = np.full_like(y_test, float(np.median(y_train)))
    model_metrics = _metrics(y_test, predictions)
    baseline_metrics = _metrics(y_test, baseline_predictions)
    residuals = y_test - predictions

    zone = clean.groupby("zone").agg(
        shipments=("shipment_id", "count"),
        mean_min=("actual_delivery_min", "mean"),
        median_min=("actual_delivery_min", "median"),
        on_time_rate=("on_time", "mean"),
        mean_cost_inr=("delivery_cost_inr", "mean"),
    ).reset_index().sort_values("on_time_rate")
    traffic = clean.groupby("traffic_level").agg(
        shipments=("shipment_id", "count"),
        mean_min=("actual_delivery_min", "mean"),
        median_min=("actual_delivery_min", "median"),
        on_time_rate=("on_time", "mean"),
    ).reindex(TRAFFIC).reset_index()
    weather = clean.groupby("weather").agg(
        shipments=("shipment_id", "count"),
        mean_min=("actual_delivery_min", "mean"),
        median_min=("actual_delivery_min", "median"),
        on_time_rate=("on_time", "mean"),
    ).reindex(WEATHER).reset_index()
    weekly = clean.assign(week=clean["order_date"].dt.to_period("W").astype(str)).groupby("week").agg(
        shipments=("shipment_id", "count"), mean_min=("actual_delivery_min", "mean"),
        on_time_rate=("on_time", "mean"),
    ).reset_index()
    stats = {
        "rows": int(len(clean)),
        "date_start": clean["order_date"].min().strftime("%d %b %Y"),
        "date_end": clean["order_date"].max().strftime("%d %b %Y"),
        "distance_mean": float(clean["distance_km"].mean()),
        "distance_median": float(clean["distance_km"].median()),
        "actual_mean": float(clean["actual_delivery_min"].mean()),
        "actual_median": float(clean["actual_delivery_min"].median()),
        "actual_sd": float(clean["actual_delivery_min"].std(ddof=1)),
        "actual_p90": float(clean["actual_delivery_min"].quantile(.90)),
        "actual_p95": float(clean["actual_delivery_min"].quantile(.95)),
        "on_time_rate": float(clean["on_time"].mean()),
        "cost_mean": float(clean["delivery_cost_inr"].mean()),
        "cost_median": float(clean["delivery_cost_inr"].median()),
        "distance_time_corr": float(clean["distance_km"].corr(clean["actual_delivery_min"])),
        "high_low_median_gap": float(
            clean.loc[clean["traffic_level"] == "High", "actual_delivery_min"].median()
            - clean.loc[clean["traffic_level"] == "Low", "actual_delivery_min"].median()
        ),
        "rain_clear_mean_gap": float(
            clean.loc[clean["weather"] == "Rain", "actual_delivery_min"].mean()
            - clean.loc[clean["weather"] == "Clear", "actual_delivery_min"].mean()
        ),
        "split_date": split_date.strftime("%d %b %Y"),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "baseline": baseline_metrics,
        "model": model_metrics,
        "coefs": dict(zip(feature_names, beta.tolist())),
        "predictions": predictions,
        "y_test": y_test,
        "residuals": residuals,
        "train_median": float(np.median(y_train)),
        "zone": zone,
        "traffic": traffic,
        "weather": weather,
        "weekly": weekly,
    }
    return stats


def _fonts():
    paths = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/Arial.ttf"]
    bold_paths = ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/Arialbd.ttf"]
    def first(paths, size):
        for path in paths:
            if Path(path).exists():
                return ImageFont.truetype(path, size=size)
        return ImageFont.load_default()
    return first(paths, 22), first(bold_paths, 30), first(paths, 17), first(bold_paths, 17)


def _canvas(title: str, subtitle: str):
    image = Image.new("RGB", (1200, 690), "white")
    draw = ImageDraw.Draw(image)
    body, title_font, small, small_bold = _fonts()
    draw.text((66, 36), title, fill="#12304A", font=title_font)
    draw.text((68, 82), subtitle, fill="#526574", font=small)
    draw.line((68, 119, 1132, 119), fill="#D9E3E8", width=2)
    return image, draw, body, small, small_bold


def _axes(draw, x0=115, y0=160, x1=1110, y1=590, y_ticks=5):
    grid = "#E5ECEF"
    for i in range(y_ticks + 1):
        y = y1 - (y1 - y0) * i / y_ticks
        draw.line((x0, y, x1, y), fill=grid, width=1)
    draw.line((x0, y0, x0, y1), fill="#738592", width=2)
    draw.line((x0, y1, x1, y1), fill="#738592", width=2)


def chart_histogram(values, mean, median, path: Path):
    image, draw, body, small, bold = _canvas("Delivery time is right-skewed", "Clean synthetic shipments | minutes | mean and median shown")
    x0, y0, x1, y1 = 130, 170, 1090, 570
    bins = np.linspace(0, max(180, np.percentile(values, 99)), 16)
    counts, edges = np.histogram(values, bins=bins)
    _axes(draw, x0, y0, x1, y1)
    draw.text((x0, y0 - 28), "Shipment count", fill="#526574", font=small)
    maximum = max(counts) or 1
    for i, count in enumerate(counts):
        left = x0 + (edges[i] / edges[-1]) * (x1 - x0)
        right = x0 + (edges[i + 1] / edges[-1]) * (x1 - x0) - 3
        top = y1 - count / maximum * (y1 - y0)
        draw.rectangle((left, top, right, y1), fill="#1B8A9B")
    for value, color, label in [(mean, "#E28A22", f"Mean {mean:.1f}"), (median, "#12304A", f"Median {median:.1f}")]:
        x = x0 + min(value / edges[-1], 1) * (x1 - x0)
        draw.line((x, y0, x, y1), fill=color, width=4)
        draw.text((x + 8, y0 + 5 if color == "#E28A22" else y0 + 35), label, fill=color, font=bold)
    draw.text((x0, 610), "0", fill="#526574", font=small)
    draw.text((x1 - 40, 610), f"{edges[-1]:.0f}", fill="#526574", font=small)
    draw.text((500, 638), "Actual delivery time (minutes)", fill="#12304A", font=body)
    image.save(path)


def chart_zone_ontime(zone: pd.DataFrame, path: Path):
    image, draw, body, small, bold = _canvas("On-time performance varies by zone", "Share delivered by the promised time | synthetic example")
    x0, x1 = 260, 1070
    top = 164
    maxw = x1 - x0
    for i, row in enumerate(zone.itertuples(index=False)):
        y = top + i * 68
        rate = float(row.on_time_rate)
        draw.text((78, y + 6), str(row.zone), fill="#12304A", font=body)
        draw.rectangle((x0, y, x1, y + 38), fill="#EEF3F5")
        draw.rectangle((x0, y, x0 + maxw * rate, y + 38), fill="#1B8A9B" if rate >= .5 else "#E28A22")
        draw.text((x0 + maxw * rate + 10, y + 7), f"{rate:.0%}  |  n={int(row.shipments)}", fill="#12304A", font=bold)
    draw.text((x0, 594), "0%", fill="#526574", font=small)
    draw.text((x1 - 42, 594), "100%", fill="#526574", font=small)
    draw.text((480, 635), "On-time delivery rate", fill="#12304A", font=body)
    image.save(path)


def chart_traffic(traffic: pd.DataFrame, path: Path):
    image, draw, body, small, bold = _canvas("Higher traffic aligns with longer delivery times", "Mean duration by traffic category | synthetic observations")
    x0, y0, x1, y1 = 145, 175, 1080, 570
    _axes(draw, x0, y0, x1, y1)
    maxval = max(traffic["mean_min"].max() * 1.2, 100)
    colors = ["#1B8A9B", "#4F8192", "#E28A22"]
    for i, row in enumerate(traffic.itertuples(index=False)):
        cx = x0 + (i + .5) * ((x1 - x0) / 3)
        h = float(row.mean_min) / maxval * (y1 - y0)
        draw.rectangle((cx - 95, y1 - h, cx + 95, y1), fill=colors[i])
        draw.text((cx - 45, y1 - h - 32), f"{row.mean_min:.1f}", fill="#12304A", font=bold)
        draw.text((cx - 45, y1 + 15), str(row.traffic_level), fill="#12304A", font=body)
    draw.text((58, y0 + 5), "min", fill="#526574", font=small)
    draw.text((500, 635), "Traffic at dispatch", fill="#12304A", font=body)
    image.save(path)


def chart_distance_scatter(clean: pd.DataFrame, path: Path):
    image, draw, body, small, bold = _canvas("Distance is associated with elapsed delivery time", "Each point is a shipment | correlation is descriptive, not causal")
    x0, y0, x1, y1 = 130, 160, 1090, 570
    _axes(draw, x0, y0, x1, y1)
    max_x = max(25.0, float(clean["distance_km"].quantile(.995)))
    max_y = max(180.0, float(clean["actual_delivery_min"].quantile(.995)))
    sample = clean.sample(min(450, len(clean)), random_state=SEED)
    for row in sample.itertuples(index=False):
        x = x0 + float(row.distance_km) / max_x * (x1 - x0)
        y = y1 - min(float(row.actual_delivery_min) / max_y, 1) * (y1 - y0)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#1B8A9B")
    draw.text((x0, 600), "0", fill="#526574", font=small)
    draw.text((x1 - 30, 600), f"{max_x:.0f} km", fill="#526574", font=small)
    draw.text((465, 638), "Route distance (km)", fill="#12304A", font=body)
    draw.text((x0, y0 - 27), f"Actual delivery time (0-{max_y:.0f} min)", fill="#526574", font=small)
    image.save(path)


def chart_weekly(weekly: pd.DataFrame, path: Path):
    image, draw, body, small, bold = _canvas("Weekly averages fluctuate across the 90-day sample", "Mean actual delivery time | shipment counts vary by week")
    x0, y0, x1, y1 = 125, 165, 1090, 565
    _axes(draw, x0, y0, x1, y1)
    draw.text((x0, y0 - 27), "Mean delivery time (min)", fill="#526574", font=small)
    vals = weekly["mean_min"].to_numpy(float)
    lo = min(45.0, float(vals.min()) - 5)
    hi = max(130.0, float(vals.max()) + 5)
    pts = []
    for i, value in enumerate(vals):
        x = x0 + (i / max(1, len(vals) - 1)) * (x1 - x0)
        y = y1 - (value - lo) / (hi - lo) * (y1 - y0)
        pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill="#1B8A9B", width=4)
    for x, y in pts:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#E28A22")
    for idx in [0, len(pts)//2, len(pts)-1]:
        if pts:
            x, y = pts[idx]
            draw.text((x - 16, y - 32), f"{vals[idx]:.0f}", fill="#12304A", font=bold)
    draw.text((x0, 595), "Early weeks", fill="#526574", font=small)
    draw.text((x1 - 88, 595), "Late weeks", fill="#526574", font=small)
    draw.text((430, 635), "Week in the sample", fill="#12304A", font=body)
    image.save(path)


def chart_prediction(y_true, y_pred, path: Path):
    image, draw, body, small, bold = _canvas("Model predictions track the broad delivery-time pattern", "Chronological holdout | actual and predicted minutes")
    x0, y0, x1, y1 = 135, 160, 1090, 570
    _axes(draw, x0, y0, x1, y1)
    limit = max(180.0, float(np.percentile(np.r_[y_true, y_pred], 99)))
    draw.line((x0, y1, x1, y0), fill="#E28A22", width=3)
    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(y_true), min(350, len(y_true)), replace=False)
    for i in picks:
        x = x0 + min(float(y_true[i]) / limit, 1) * (x1 - x0)
        y = y1 - min(float(y_pred[i]) / limit, 1) * (y1 - y0)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#1B8A9B")
    draw.text((x0, 600), "0", fill="#526574", font=small)
    draw.text((x1 - 36, 600), f"{limit:.0f}", fill="#526574", font=small)
    draw.text((460, 638), "Actual delivery time (minutes)", fill="#12304A", font=body)
    draw.text((22, 350), "Predicted", fill="#526574", font=small)
    image.save(path)


def chart_permutation_importance(clean: pd.DataFrame, result: dict, path: Path):
    image, draw, body, small, bold = _canvas("Distance and traffic carry the most predictive signal", "Increase in holdout MAE after shuffling one input | minutes")
    split_date = pd.Timestamp(result["split_date"])
    dates = sorted(clean["order_date"].unique())
    # Recover the exact original split from the displayed split date.
    train = clean.loc[clean["order_date"] < split_date]
    test = clean.loc[clean["order_date"] >= split_date].copy()
    x_train, names, means, stds = _feature_matrix(train)
    beta = np.array([result["coefs"][name] for name in names])
    y_test = test["actual_delivery_min"].to_numpy(float)
    x_test, _, _, _ = _feature_matrix(test, means, stds)
    base_mae = np.mean(np.abs(y_test - x_test @ beta))
    importances = []
    rng = np.random.default_rng(SEED + 1)
    for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        shuffled = test.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        x_shuffled, _, _, _ = _feature_matrix(shuffled, means, stds)
        importances.append((feature, float(np.mean(np.abs(y_test - x_shuffled @ beta)) - base_mae)))
    importances.sort(key=lambda item: item[1], reverse=True)
    top = importances[:6]
    x0, x1 = 300, 1070
    maxval = max(max(v for _, v in top), 1.0)
    for i, (name, value) in enumerate(top):
        y = 157 + i * 65
        draw.text((65, y + 8), name.replace("_", " ").title(), fill="#12304A", font=body)
        draw.rectangle((x0, y, x1, y + 40), fill="#EEF3F5")
        w = max(0.0, value) / maxval * (x1 - x0)
        draw.rectangle((x0, y, x0 + w, y + 40), fill="#1B8A9B" if i else "#E28A22")
        draw.text((x0 + w + 8, y + 8), f"{value:.1f}", fill="#12304A", font=bold)
    draw.text((438, 620), "Holdout MAE increase (minutes)", fill="#12304A", font=body)
    image.save(path)
    result["permutation_importance"] = importances


def make_charts(clean: pd.DataFrame, result: dict, directory: Path = CHART_DIR) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "histogram": directory / "delivery_time_distribution.png",
        "zone": directory / "on_time_by_zone.png",
        "traffic": directory / "duration_by_traffic.png",
        "distance": directory / "distance_vs_delivery_time.png",
        "weekly": directory / "weekly_delivery_trend.png",
        "prediction": directory / "actual_vs_predicted.png",
        "importance": directory / "model_feature_importance.png",
    }
    chart_histogram(clean["actual_delivery_min"].to_numpy(float), result["actual_mean"], result["actual_median"], paths["histogram"])
    chart_zone_ontime(result["zone"], paths["zone"])
    chart_traffic(result["traffic"], paths["traffic"])
    chart_distance_scatter(clean, paths["distance"])
    chart_weekly(result["weekly"], paths["weekly"])
    chart_prediction(result["y_test"], result["predictions"], paths["prediction"])
    chart_permutation_importance(clean, result, paths["importance"])
    return paths


def main():
    raw = build_raw_data()
    clean, counts = clean_data(raw)
    raw.to_csv(ROOT / "synthetic_pune_last_mile_shipments_raw.csv", index=False, date_format="%Y-%m-%d")
    output = ROOT / "synthetic_pune_last_mile_shipments.csv"
    clean.to_csv(output, index=False, date_format="%Y-%m-%d")
    result = analyze(clean)
    make_charts(clean, result)
    print(f"Raw rows: {counts['raw_rows']}; duplicate rows: {counts['duplicate_rows']}; clean rows: {counts['clean_rows']}")
    print(f"Dates: {result['date_start']} to {result['date_end']}; on-time: {result['on_time_rate']:.1%}")
    print(f"Delivery minutes mean={result['actual_mean']:.1f}, median={result['actual_median']:.1f}, p90={result['actual_p90']:.1f}")
    print(f"Baseline: MAE={result['baseline']['mae']:.2f}, RMSE={result['baseline']['rmse']:.2f}, R2={result['baseline']['r2']:.3f}")
    print(f"Linear model: MAE={result['model']['mae']:.2f}, RMSE={result['model']['rmse']:.2f}, R2={result['model']['r2']:.3f}")
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()
