"""
synth/structural.py

Structural theme profiles, context configurations, and specialized matrix generation settings
for heatmaps and advanced coordinated synthetic visualizations.
Encapsulates structural generative themes outside of visual styling (themes.py).
"""
from typing import Any, Dict, List, Tuple

# ===================================================================================
# == CONTEXT-AWARE VISUAL CONFIGURATIONS FOR HEATMAPS ==
# ===================================================================================

CONTEXT_CONFIGURATIONS: Dict[str, Dict[str, Any]] = {
    "genomic_expression_heatmap": {
        "domain": "scientific",
        "heatmap_type": "bicluster_checkerboard",
        "colormap": "coolwarm",
        "normalization_strategy": "row_z_score",
        "cbar_label": "Standardized Expression (Z-Score)",
        "annotation_format": None,
        "xlabel_pool": ["Gene Symbol", "Transcript ID", "Pathway"],
        "ylabel_pool": ["Patient Cohort", "Treatment Arm", "Sample Group"],
        "title_pool": ["Gene Expression Heatmap", "Treatment Effect Profile"]
    },
    "cohort_retention_heatmap": {
        "domain": "business",
        "heatmap_type": "clustered_patterns",
        "colormap": "YlGnBu",
        "normalization_strategy": "global_min_max",
        "cbar_label": "Net Retention Rate (%)",
        "annotation_format": "{:.0f}%",
        "xlabel_pool": ["Acquisition Cohort", "Billing Cycle"],
        "ylabel_pool": ["Customer Segment", "Plan Tier"],
        "title_pool": ["Cohort Retention Matrix", "Retention Cohort Heatmap"]
    },
    "correlation_matrix": {
        "domain": "scientific",
        "heatmap_type": "correlation_lkj",
        "colormap": "RdBu_r",
        "normalization_strategy": "fixed_bounds_pm_1",
        "cbar_label": "Pearson Correlation Coefficient",
        "annotation_format": "{:.2f}",
        "xlabel_pool": ["Feature", "Assay", "Metric"],
        "ylabel_pool": ["Feature", "Assay", "Metric"],
        "title_pool": ["Correlation Matrix", "Similarity Index Map"]
    },
    "pharmacokinetics_heatmap": {
        "domain": "scientific",
        "heatmap_type": "sarima",
        "colormap": "viridis",
        "normalization_strategy": "log10_scale",
        "cbar_label": "Plasma Concentration (ng/mL)",
        "annotation_format": None,
        "xlabel_pool": ["Time Post-dose (h)", "Sampling Time (min)"],
        "ylabel_pool": ["Patient Cohort", "Dose Group"],
        "title_pool": ["Pharmacokinetic Heatmap", "Exposure Over Time"]
    }
}

# ===================================================================================
# == STRUCTURAL THEMES ==
# ===================================================================================

STRUCTURAL_THEMES: Dict[str, Dict[str, Any]] = {
    "biclustered_checkerboard": {
        "description": "Localized coordinated features (checkerboard biclusters).",
        "backend_function": "sklearn.datasets.make_checkerboard",
        "generation_parameters": {
            "n_clusters": (5, 4),
            "noise": 12.5,
            "minval": -50.0,
            "maxval": 50.0,
            "shuffle": True
        },
        "validation_metric": "Spectral Co-clustering Recovery Accuracy"
    },
    "block_diagonal_communities": {
        "description": "Independent block communities with sparse cross-talk.",
        "backend_function": "scipy.linalg.block_diag",
        "generation_parameters": {
            "num_blocks": 8,
            "block_size_bounds": (15, 60),
            "off_diagonal_noise_std": 0.15,
            "internal_block_density": 0.90
        },
        "validation_metric": "RMSR < 0.05"
    },
    "toeplitz_autoregressive": {
        "description": "Autoregressive decay structure for temporal proximity.",
        "backend_function": "scipy.linalg.toeplitz",
        "generation_parameters": {
            "decay_coefficient": 0.88,
            "dimensions": 150
        },
        "validation_metric": "DensityCut Stability Measure"
    }
}

# ===================================================================================
# == HEATMAP ANNOTATION FORMATS & COLORBAR TITLES ==
# ===================================================================================

HEATMAP_ANNOTATION_FORMATS: Tuple[str, ...] = (
    "{:.2f}",  # Two decimal places
    "{:.1f}",  # One decimal place
    "{:.3f}",  # Three decimal places
    "{:.0f}",  # Integer
    "{:.1%}",  # Percentage with one decimal
    "{:.0%}",  # Percentage integer
    "{:.2e}",  # Scientific notation
)

COLORBAR_TITLES_SCIENTIFIC: Tuple[str, ...] = (
    "Expression Level (log₂)",
    "Normalized Intensity (z-score)",
    "Fold Change (log₂ FC)",
    "Correlation Coefficient (r)",
    "p-value (-log₁₀)",
    "Percent Change (%)",
    "Relative Abundance",
    "Signal-to-Noise Ratio",
    "Effect Size (Cohen's d)",
    "Normalized Response",
    "Activity Score (0-1)",
    "Enrichment Score",
    "Similarity Index",
    "Distance Metric",
    "Probability Density",
    "Rate Constant (s⁻¹)",
    "Concentration (μM)",
    "Intensity (a.u.)",
    "Fluorescence (RFU)",
    "Absorbance (OD)",
    "Coverage (%)", "Read Depth (reads)", "Methylation (%)", "Beta Value",
    "Counts per Million (CPM)", "TPM", "RPKM", "Delta Ct", "Ct Value",
    "Log Odds", "Binding Affinity (K_D, nM)", "Enrichment (NES)",
    "Adjusted p-value (q)", "Probability (0-1)", "Likelihood Ratio",
    "Velocity (mm/s)", "Frequency (Hz)", "Optical Density (AU)",
    "Normalized Counts", "Signal Intensity (counts)"
)

COLORBAR_TITLES_BUSINESS: Tuple[str, ...] = (
    "Revenue ($M)",
    "Growth Rate (%)",
    "Market Share (%)",
    "Customer Satisfaction (1-10)",
    "Net Promoter Score",
    "Conversion Rate (%)",
    "Engagement Score",
    "Performance Index",
    "Efficiency Ratio",
    "Return on Investment (%)",
    "Cost per Acquisition ($)",
    "Lifetime Value ($K)",
    "Churn Probability (%)",
    "Inventory Turnover",
    "Fill Rate (%)",
    "Response Time (min)",
    "Utilization Rate (%)",
    "Quality Score (0-100)",
    "Average Order Value ($)", "ARPU ($)", "MRR ($)",
    "Gross Margin (%)", "Operating Margin (%)", "EBITDA ($M)",
    "Days Sales Outstanding (DSO)", "Customer Retention Rate (%)",
    "CAC Payback (months)", "Cart Abandonment Rate (%)",
    "Return on Ad Spend (ROAS)", "Cost of Goods Sold ($)",
    "Payroll Expense ($K)", "Revenue per Employee ($K)",
    "Inventory Days", "Defect Rate (%)", "On-time Delivery (%)",
    "First Response Time (min)", "SLA Compliance (%)", "Fraud Rate (%)"
)

HEATMAP_CHART_TITLES: Tuple[str, ...] = (
    "Correlation Matrix",
    "Gene Expression Heatmap",
    "Sample Clustering Analysis",
    "Time-Course Response Pattern",
    "Treatment Effect Profile",
    "Cross-Feature Association Map",
    "Hierarchical Clustering Result",
    "Distance Matrix Visualization",
    "Temporal Activity Pattern",
    "Multi-Variable Comparison",
    "Performance Scorecard",
    "Regional Sales Performance",
    "Customer Behavior Matrix",
    "Product Performance Comparison",
    "Cohort Analysis Dashboard",
    "A/B Test Results Matrix",
    "Channel Attribution Heatmap",
    "Seasonal Trend Analysis",
    "Risk Assessment Matrix",
    "Quality Control Dashboard",
    "Supply Chain Heatmap", "Inventory Level Over Time",
    "Sales Funnel Conversion Matrix", "Customer Lifetime Value Map",
    "Employee Performance Matrix", "Service Latency Heatmap",
    "Model Confusion Matrix", "Hyperparameter Sweep Results",
    "Feature Importance Matrix", "Anomaly Detection Heatmap",
    "Genotype-Phenotype Association Map", "Protein-Protein Interaction Map",
    "Metabolomics Profile Heatmap", "Pharmacokinetic Heatmap",
    "Electrophysiology Activity Map", "Brain Activation Map",
    "Operational Risk Heatmap", "Fraud Detection Score Matrix",
    "Retention Cohort Heatmap", "Market Penetration Map",
    "Channel vs Product Performance"
)

HEATMAP_XLABELS_SCIENTIFIC: Tuple[str, ...] = (
    "Sample ID", "Replicate Number", "Time Point (h)", "Time Point (min)",
    "Treatment Dose (μg/mL)", "Dose Level (nM)", "Concentration (μM)",
    "Patient Cohort", "Experimental Condition", "Cell Line", "Tissue Type",
    "Gene Symbol", "Protein ID", "Metabolite", "Compound ID",
    "Observation Day", "Culture Passage", "Treatment Duration (h)",
    "Temperature (°C)", "pH Level", "Stimulus Intensity",
    "Wavelength (nm)", "Frequency (Hz)", "Channel Number",
    "Experiment ID", "Plate Number", "Well ID", "Barcode", "Batch ID",
    "Run ID", "Instrument ID", "Operator", "Library Prep Kit",
    "Sequencing Depth (reads)", "Read Length (bp)", "Coverage (%)",
    "Library Batch", "Flowcell ID", "Lane Number", "Probe ID",
    "Antibody Clone", "Staining Protocol", "Microscope Objective",
    "Magnification (x)", "Pixel Size (μm)", "ROI ID", "Electrode ID",
    "Stimulus Frequency (Hz)", "Inoculation Route", "Dosage (mg/kg)",
    "Subject ID", "Age (years)", "Sex", "BMI (kg/m²)", "Ethnicity",
    "Comorbidity Flag", "Clinical Site", "Trial Arm", "Visit Number",
    "Post-dose Time (min)", "Pre-treatment Baseline", "Control Group",
    "Treatment Group", "Storage Condition", "Freeze-Thaw Cycles",
    "Extraction Method", "Assay Kit Lot", "Instrument Channel",
    "Sensor ID", "Acquisition Mode", "Imaging Modality"
)

HEATMAP_YLABELS_SCIENTIFIC: Tuple[str, ...] = (
    "Gene Symbol", "Protein Target", "Transcript ID", "Pathway",
    "Cellular Component", "Biological Process", "Metabolite",
    "Biomarker", "Clinical Parameter", "Measurement Type",
    "Assay Replicate", "Technical Replicate", "Sample Group",
    "Treatment Arm", "Disease Stage", "Severity Score",
    "Feature Vector", "Principal Component", "Latent Factor",
    "Response Variable", "Outcome Measure",
    "Gene Ontology Term", "Enzyme", "Reaction ID", "Metabolic Pathway",
    "SNP ID", "CpG Site", "Cell Type", "Subcellular Location",
    "Isoform", "Transcript Variant", "Protein Domain", "Protein Motif",
    "Binding Site", "Epitope", "Post-translational Mod", "Phosphosite",
    "Interaction Partner", "Protein Complex", "Phenotype",
    "Survival Time (days)", "Hazard Ratio", "Adverse Event", "IC50 (nM)",
    "EC50 (μM)", "LOD (a.u.)", "LOQ", "Signal-to-Background",
    "Normalized Count (TPM)", "RPKM", "CPM", "Delta Ct", "Ct Value",
    "Z-score", "t-statistic", "Adjusted p-value (FDR)", "Enrichment Term",
    "Cluster Label", "Module ID", "Annotation", "Taxon ID", "Strain"
)

HEATMAP_XLABELS_BUSINESS: Tuple[str, ...] = (
    "Product SKU", "Sales Region", "Customer Segment", "Marketing Channel",
    "Campaign ID", "Fiscal Quarter", "Business Week", "Month",
    "Store Location", "Distribution Center", "Sales Territory",
    "Customer Cohort", "Account Type", "Service Tier",
    "Device Type", "Platform", "Acquisition Source",
    "Order ID", "Invoice Period", "Sales Rep", "Promotion Type",
    "Price Tier", "List Price ($)", "Discount (%)", "Bundle ID",
    "Customer Lifetime Stage", "Subscription Plan", "Billing Cycle",
    "Payment Method", "Fulfillment Channel", "Return Reason",
    "Lead Source", "Landing Page", "Ad Group", "Keyword",
    "Traffic Channel", "Session Source", "Session Medium", "Browser",
    "Operating System", "App Version", "Feature Flag", "Product Line",
    "Product Family", "Manufacturing Batch", "Supplier ID", "PO Number",
    "Shipping Method", "Delivery Window", "Stock Keeping Unit",
    "Warehouse Zone", "Shelf Location", "CSR Team", "Contract Type"
)

HEATMAP_YLABELS_BUSINESS: Tuple[str, ...] = (
    "Key Performance Indicator", "Revenue Stream", "Cost Center",
    "Product Category", "Service Line", "Business Unit",
    "Performance Metric", "Engagement Metric", "Conversion Stage",
    "Customer Attribute", "Behavioral Segment", "Risk Category",
    "Quality Metric", "Operational KPI", "Financial Ratio",
    "Customer Lifetime Value", "Average Order Value", "Churn Rate",
    "Retention Cohort", "Funnel Stage", "Lead Quality", "Sales Velocity",
    "Support Ticket Type", "SLA Breach", "Inventory Level",
    "Days of Inventory", "Stockout Indicator", "Supplier Performance",
    "On-time Delivery Rate", "Return Rate", "Warranty Claims",
    "Defect Rate", "Cycle Time", "Throughput", "Downtime Minutes",
    "Utilization", "Headcount", "Cost per Employee", "Training Hours",
    "Compliance Flag", "Risk Score", "Fraud Indicator", "Net Revenue",
    "Gross Margin", "Operating Expense Category"
)
