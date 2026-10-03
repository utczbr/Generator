"""
synth/tabular.py

Lightweight multivariate tabular modeling layer using Gaussian copulas
and native Cholesky decomposition (Sigma = L L^T) to simulate correlated series
with strictly bounded marginal distributions and domain-consistent semantics.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats

from synth.semantics.enums import UnknownDomainError


DOMAIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "biomedical": {
        "x_labels": [
            "Dose (mg/kg)",
            "Concentration (μM)",
            "Time Post-Dose (h)",
            "Treatment Cohort",
            "Patient Group",
        ],
        "categories_pool": [
            ["Control", "Vehicle", "Low Dose", "Mid Dose", "High Dose"],
            ["Wild Type", "Mutant A", "Mutant B", "Knockout"],
            ["Placebo", "Dose 10mg", "Dose 25mg", "Dose 50mg"],
            ["Baseline", "Week 2", "Week 4", "Week 8"],
        ],
        "metrics": [
            {"name": "Plasma Concentration (ng/mL)", "dist": "lognormal", "params": {"s": 0.4, "scale": 120.0}, "bounds": (0.0, None)},
            {"name": "Cell Viability (%)", "dist": "beta", "params": {"a": 5.0, "b": 2.0, "scale": 100.0}, "bounds": (0.0, 100.0)},
            {"name": "Inhibition (%)", "dist": "beta", "params": {"a": 2.0, "b": 5.0, "scale": 100.0}, "bounds": (0.0, 100.0)},
            {"name": "AUC (ng·h/mL)", "dist": "lognormal", "params": {"s": 0.5, "scale": 450.0}, "bounds": (0.0, None)},
            {"name": "Enzyme Activity (U/mL)", "dist": "lognormal", "params": {"s": 0.3, "scale": 85.0}, "bounds": (0.0, None)},
        ],
        "correlation_pairs": {
            ("Cell Viability (%)", "Inhibition (%)"): -0.82,
            ("Plasma Concentration (ng/mL)", "AUC (ng·h/mL)"): 0.86,
        },
        "titles": [
            "Dose–Response Curve",
            "Pharmacokinetic Exposure Profile",
            "Cell Viability Across Treatment Arms",
            "Enzyme Kinetics and Inhibition Study",
            "Clinical Biomarker Response",
        ],
    },
    "business": {
        "x_labels": [
            "Fiscal Quarter",
            "Business Unit",
            "Geographic Region",
            "Customer Tier",
            "Product Category",
        ],
        "categories_pool": [
            ["Q1", "Q2", "Q3", "Q4"],
            ["North America", "EMEA", "APAC", "LATAM"],
            ["Enterprise", "Mid-Market", "Commercial", "SMB"],
            ["Hardware", "Software", "Cloud Services", "Support"],
        ],
        "metrics": [
            {"name": "Revenue ($M)", "dist": "lognormal", "params": {"s": 0.4, "scale": 80.0}, "bounds": (0.0, None)},
            {"name": "Gross Profit ($M)", "dist": "lognormal", "params": {"s": 0.4, "scale": 50.0}, "bounds": (0.0, None)},
            {"name": "Operating Expenses ($M)", "dist": "lognormal", "params": {"s": 0.35, "scale": 30.0}, "bounds": (0.0, None)},
            {"name": "Customer Acquisition Cost ($)", "dist": "lognormal", "params": {"s": 0.3, "scale": 1200.0}, "bounds": (0.0, None)},
            {"name": "Customer Lifetime Value ($)", "dist": "lognormal", "params": {"s": 0.35, "scale": 4500.0}, "bounds": (0.0, None)},
        ],
        "correlation_pairs": {
            ("Revenue ($M)", "Gross Profit ($M)"): 0.85,
            ("Customer Acquisition Cost ($)", "Customer Lifetime Value ($)"): 0.72,
        },
        "titles": [
            "Quarterly Revenue & Profitability Analysis",
            "Business Unit Financial Performance",
            "Operating Expense vs Gross Margin Breakdown",
            "Customer Acquisition Cost vs Lifetime Value",
            "Regional Sales and EBITDA Overview",
        ],
    },
    "demographic": {
        "x_labels": [
            "Age Cohort",
            "Education Level",
            "Metropolitan Area",
            "Income Bracket",
            "Survey Wave",
        ],
        "categories_pool": [
            ["18-24", "25-34", "35-49", "50-64", "65+"],
            ["High School", "Associate", "Bachelor's", "Master's", "Doctorate"],
            ["Under $30k", "$30k-$60k", "$60k-$100k", "$100k-$150k", "$150k+"],
            ["Northeast", "Midwest", "South", "West"],
        ],
        "metrics": [
            {"name": "Median Household Income ($k)", "dist": "lognormal", "params": {"s": 0.3, "scale": 72.0}, "bounds": (0.0, None)},
            {"name": "Employment Rate (%)", "dist": "beta", "params": {"a": 8.0, "b": 2.0, "scale": 100.0}, "bounds": (0.0, 100.0)},
            {"name": "Homeownership Rate (%)", "dist": "beta", "params": {"a": 6.0, "b": 3.0, "scale": 100.0}, "bounds": (0.0, 100.0)},
            {"name": "Higher Education Attainment (%)", "dist": "beta", "params": {"a": 4.0, "b": 4.0, "scale": 100.0}, "bounds": (0.0, 100.0)},
        ],
        "correlation_pairs": {
            ("Median Household Income ($k)", "Higher Education Attainment (%)"): 0.76,
            ("Employment Rate (%)", "Homeownership Rate (%)"): 0.58,
        },
        "titles": [
            "Demographic Trends by Age Cohort",
            "Education Level vs Median Income Distribution",
            "Regional Homeownership and Employment Study",
            "Metropolitan Population & Economic Indicators",
        ],
    },
    "engineering": {
        "x_labels": [
            "Timestamp (s)",
            "Operational Cycle",
            "Turbine Unit",
            "Sensor Channel",
            "Sampling Interval (min)",
        ],
        "categories_pool": [
            ["Turbine 1", "Turbine 2", "Turbine 3", "Turbine 4"],
            ["Channel A", "Channel B", "Channel C", "Channel D"],
            ["Idle", "Ramp-up", "Steady State", "Peak Load"],
            ["Stage 1", "Stage 2", "Stage 3", "Stage 4"],
        ],
        "metrics": [
            {"name": "Temperature (°C)", "dist": "normal", "params": {"loc": 65.0, "scale": 12.0}, "bounds": (0.0, 150.0)},
            {"name": "Vibration (mm/s)", "dist": "lognormal", "params": {"s": 0.35, "scale": 4.5}, "bounds": (0.0, None)},
            {"name": "Pressure (bar)", "dist": "lognormal", "params": {"s": 0.25, "scale": 18.0}, "bounds": (0.0, None)},
            {"name": "Power Consumption (kW)", "dist": "lognormal", "params": {"s": 0.3, "scale": 250.0}, "bounds": (0.0, None)},
            {"name": "Thermal Efficiency (%)", "dist": "beta", "params": {"a": 7.0, "b": 2.0, "scale": 100.0}, "bounds": (0.0, 100.0)},
        ],
        "correlation_pairs": {
            ("Power Consumption (kW)", "Temperature (°C)"): 0.81,
            ("Pressure (bar)", "Temperature (°C)"): 0.65,
        },
        "titles": [
            "Turbine Telemetry and Vibration Profile",
            "Power Consumption vs Operating Temperature",
            "Sensor Diagnostic Monitoring Over Time",
            "Industrial Subsystem Pressure and Speed Metrics",
        ],
    },
}


def sample_gaussian_copula(
    correlation_matrix: np.ndarray,
    num_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample uniform marginals U in (0, 1) with Gaussian copula dependence
    using Cholesky factorization Sigma = L L^T.
    Returns array U of shape (k, num_samples).
    """
    if rng is None:
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))

    k = correlation_matrix.shape[0]
    R = (correlation_matrix + correlation_matrix.T) / 2.0
    np.fill_diagonal(R, 1.0)

    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        # Regularize to nearest positive definite matrix
        eigvals, eigvecs = np.linalg.eigh(R)
        eigvals = np.maximum(eigvals, 1e-4)
        R_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
        d = np.sqrt(np.diag(R_reg))
        R_reg = R_reg / np.outer(d, d)
        L = np.linalg.cholesky(R_reg)

    Z = rng.standard_normal(size=(k, num_samples))
    X = L @ Z
    # Probability integral transform: Phi(X) ~ Uniform(0, 1)
    # Clip to avoid exact 0 or 1 in inverse CDF evaluation
    U = np.clip(stats.norm.cdf(X), 1e-6, 1.0 - 1e-6)
    return U


def transform_to_marginal(
    u: np.ndarray,
    dist_type: str,
    params: Dict[str, Any],
    bounds: Tuple[Optional[float], Optional[float]] = (None, None),
) -> np.ndarray:
    """
    Transform uniform variate u in (0, 1) to target marginal distribution via inverse CDF (PPF).
    Non-negativity and bounds are preserved analytically.
    """
    if dist_type == "lognormal":
        s = params.get("s", 0.5)
        scale = params.get("scale", 100.0)
        loc = params.get("loc", 0.0)
        y = stats.lognorm.ppf(u, s=s, loc=loc, scale=scale)
    elif dist_type == "beta":
        a = params.get("a", 2.0)
        b = params.get("b", 5.0)
        scale = params.get("scale", 100.0)
        loc = params.get("loc", 0.0)
        y = stats.beta.ppf(u, a=a, b=b, loc=loc, scale=scale)
    elif dist_type == "normal":
        loc = params.get("loc", 50.0)
        scale = params.get("scale", 10.0)
        y = stats.norm.ppf(u, loc=loc, scale=scale)
    elif dist_type == "uniform":
        loc = params.get("loc", 0.0)
        scale = params.get("scale", 100.0)
        y = stats.uniform.ppf(u, loc=loc, scale=scale)
    else:
        # Default fallback to scaled normal
        y = stats.norm.ppf(u, loc=50.0, scale=15.0)

    # If lower bound is specified and distribution is unbounded below (e.g. normal),
    # strictly shift/wrap or handle bounds gracefully
    min_b, max_b = bounds
    if min_b is not None and np.any(y < min_b):
        y = np.maximum(y, min_b)
    if max_b is not None and np.any(y > max_b):
        y = np.minimum(y, max_b)

    return y


def sample_domain_schema(
    domain: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Sample a coherent domain preset schema ensuring zero cross-domain bleeding."""
    if rng is None:
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))

    if domain is not None:
        if domain not in DOMAIN_PRESETS:
            raise UnknownDomainError(
                f"Unknown or unsupported domain '{domain}' for tabular synthesis. "
                f"Supported tabular presets: {sorted(DOMAIN_PRESETS.keys())}"
            )
        selected_domain = domain
    else:
        selected_domain = str(rng.choice(list(DOMAIN_PRESETS.keys())))

    preset = DOMAIN_PRESETS[selected_domain]

    x_label = str(rng.choice(preset["x_labels"]))
    title = str(rng.choice(preset["titles"]))
    cat_idx = int(rng.integers(0, len(preset["categories_pool"])))
    categories = list(preset["categories_pool"][cat_idx])

    return {
        "domain": selected_domain,
        "x_label": x_label,
        "title": title,
        "categories": categories,
        "metrics": list(preset["metrics"]),
        "correlation_pairs": dict(preset.get("correlation_pairs", {})),
    }


def sample_coherent_labels(
    domain: Optional[str] = None,
    is_scientific: Optional[bool] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Sample coherent axis titles and chart metadata from a single domain.
    Prevents cross-domain bleeding (e.g. pharmacokinetic units on fiscal quarters).
    """
    if rng is None:
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))

    if domain is None:
        if is_scientific is True:
            domain = str(rng.choice(["biomedical", "engineering"]))
        elif is_scientific is False:
            domain = "business"
        else:
            domain = str(rng.choice(list(DOMAIN_PRESETS.keys())))

    schema = sample_domain_schema(domain=domain, rng=rng)
    metrics = schema["metrics"]
    metric_names = [m["name"] for m in metrics]

    return {
        "domain": schema["domain"],
        "x_label": schema["x_label"],
        "y_label": metric_names[0] if metric_names else "Value",
        "y_labels": metric_names,
        "title": schema["title"],
        "categories": schema["categories"],
    }


def sample_multivariate_table(
    num_rows: int,
    num_series: int = 2,
    domain: Optional[str] = None,
    correlation_spec: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Generate a correlated multivariate tabular dataset with physical bounds and domain coherence.
    Returns:
      {
        "domain": domain_name,
        "x_label": x_axis_title,
        "y_label": primary_y_axis_title,
        "title": chart_title,
        "categories": category_names,
        "series_names": [name_1, ...],
        "data": np.ndarray of shape (num_rows, num_series),
        "correlation_target": R,
        "correlation_empirical": empirical_corr,
      }
    """
    if rng is None:
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))

    schema = sample_domain_schema(domain=domain, rng=rng)
    avail_metrics = schema["metrics"]

    # Select num_series metrics from the domain
    if len(avail_metrics) >= num_series:
        chosen_metrics = list(rng.choice(avail_metrics, size=num_series, replace=False))
    else:
        chosen_metrics = list(avail_metrics)
        while len(chosen_metrics) < num_series:
            chosen_metrics.append(dict(avail_metrics[len(chosen_metrics) % len(avail_metrics)]))

    # Build target correlation matrix
    if correlation_spec is not None and correlation_spec.shape == (num_series, num_series):
        R = correlation_spec.copy()
    else:
        R = np.eye(num_series)
        corr_pairs = schema["correlation_pairs"]
        for i in range(num_series):
            for j in range(i + 1, num_series):
                m_i = chosen_metrics[i]["name"]
                m_j = chosen_metrics[j]["name"]
                rho = corr_pairs.get((m_i, m_j), corr_pairs.get((m_j, m_i), None))
                if rho is None:
                    # Default moderate positive correlation for series in same domain
                    rho = 0.65
                R[i, j] = rho
                R[j, i] = rho

    # Draw uniform variates with Gaussian copula
    U = sample_gaussian_copula(R, num_rows, rng=rng)

    # Transform to marginal distributions via inverse CDF
    data_cols = []
    for i in range(num_series):
        metric = chosen_metrics[i]
        col = transform_to_marginal(
            U[i],
            dist_type=metric["dist"],
            params=metric["params"],
            bounds=metric.get("bounds", (None, None)),
        )
        data_cols.append(col)

    data = np.column_stack(data_cols)

    # Build categories matching num_rows
    base_cats = schema["categories"]
    if len(base_cats) >= num_rows:
        categories = base_cats[:num_rows]
    else:
        prefix = schema["x_label"].split()[0]
        categories = [f"{prefix} {k+1}" for k in range(num_rows)]

    series_names = [m["name"] for m in chosen_metrics]
    empirical_corr = np.corrcoef(data.T) if num_rows > 1 else np.eye(num_series)

    return {
        "domain": schema["domain"],
        "x_label": schema["x_label"],
        "y_label": series_names[0] if series_names else "Value",
        "title": schema["title"],
        "categories": categories,
        "series_names": series_names,
        "data": data,
        "correlation_target": R,
        "correlation_empirical": empirical_corr,
    }


def sample_dose_response_series(
    num_points: int = 15,
    num_series: int = 2,
    domain: str = "biomedical",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Generate correlated multi-series dose-response curves with realistic
    Hill slopes, EC50 shifts, and heteroscedastic noise.
    """
    if rng is None:
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))

    # Concentration range: 10^-9 to 10^-4 M
    log_conc = np.linspace(-9.0, -4.0, num_points)
    doses = 10.0 ** log_conc

    # Sample correlated EC50 and baseline parameters
    corr = 0.70
    R = np.full((num_series, num_series), corr)
    np.fill_diagonal(R, 1.0)
    U = sample_gaussian_copula(R, 1, rng=rng)[:, 0]

    curves = []
    series_names = []
    for s in range(num_series):
        # Inverse CDF for EC50 in [-8.0, -5.0]
        ec50 = float(stats.uniform.ppf(U[s], loc=-8.0, scale=3.0))
        hill_slope = float(rng.uniform(0.9, 1.8))
        baseline = float(rng.uniform(2.0, 8.0))
        max_response = float(rng.uniform(85.0, 98.0))

        # Hill sigmoid equation
        response = baseline + (max_response - baseline) / (1.0 + 10.0 ** ((ec50 - log_conc) * hill_slope))

        # Heteroscedastic noise with positive constraint via LogNormal PPF
        norm_resp = np.clip((response - baseline) / max(1.0, max_response - baseline), 0.01, 0.99)
        cv = 0.04 + 0.08 * np.sqrt(norm_resp * (1.0 - norm_resp))
        noise_u = rng.uniform(0.01, 0.99, size=num_points)
        multiplicative_noise = stats.lognorm.ppf(noise_u, s=cv, scale=1.0)
        response_noisy = response * multiplicative_noise

        curves.append(response_noisy)
        series_names.append(f"Compound {chr(65 + s)}" if s < 26 else f"Series {s+1}")

    data = np.column_stack(curves)
    return {
        "domain": domain,
        "x_label": "Concentration (log M)",
        "y_label": "Response (%)",
        "title": "Dose–Response Curve",
        "doses": doses,
        "log_conc": log_conc,
        "series_names": series_names,
        "data": data,
    }
