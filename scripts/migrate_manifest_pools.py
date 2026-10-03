"""
scripts/migrate_manifest_pools.py

One-time migration script for SEM-03 / SEM-08:
- Adds explicit `pools:` to all metrics based on current index boundaries:
  - common: 0..39 -> ["canonical_independent"], 40..90 -> ["histogram_y"]
  - biomedical: 0..74 -> ["legacy"]
  - engineering: 0..102 -> ["legacy"]
  - business: 0..85 -> ["legacy"]
- Adds manifest_version: 2, default_fallbacks, and capabilities to all manifests (§3.3).
Preserves YAML comments and formatting using ruamel.yaml.
"""
from pathlib import Path
from ruamel.yaml import YAML

DOMAINS_DIR = Path("synth/semantics/domains")

FALLBACKS = {
    "biomedical": ["Time (h)", "Concentration (μM)"],
    "engineering": ["Time (s)", "Temperature (°C)"],
    "business": ["Quarter", "Revenue ($M)"],
    "demographic": ["Age Cohort", "Employment Rate (%)"],
}

CAPABILITIES = {
    "biomedical": {"tabular_preset": True, "heatmap_pools": True},
    "engineering": {"tabular_preset": True, "heatmap_pools": True},
    "business": {"tabular_preset": True, "heatmap_pools": True},
    "demographic": {"tabular_preset": False, "heatmap_pools": False},
    "common": {"tabular_preset": False, "heatmap_pools": False},
}

DEFAULT_WEIGHTS = {
    "biomedical": 0.70,
    "engineering": 0.30,
    "business": 0.80,
    "demographic": 0.20,
    "common": 1.0,
}

ORDERS = {
    "biomedical": 1,
    "engineering": 2,
    "business": 3,
    "demographic": 4,
    "common": 10,
}


def migrate():
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=2, offset=0)

    for domain_name in ("common", "biomedical", "engineering", "business", "demographic"):
        yaml_path = DOMAINS_DIR / f"{domain_name}.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.load(f)

        data["manifest_version"] = 2
        data["order"] = ORDERS.get(domain_name, 50)
        data["default_weight"] = DEFAULT_WEIGHTS.get(domain_name, 1.0)
        if domain_name in FALLBACKS:
            data["default_fallbacks"] = FALLBACKS[domain_name]
        data["capabilities"] = CAPABILITIES.get(domain_name, {"tabular_preset": False, "heatmap_pools": False})

        metrics = data.get("metrics", [])
        for idx, m in enumerate(metrics):
            pools = []
            if domain_name == "common":
                if idx < 40:
                    pools.append("canonical_independent")
                else:
                    pools.append("histogram_y")
            elif domain_name == "biomedical" and idx < 75:
                pools.append("legacy")
            elif domain_name == "engineering" and idx < 103:
                pools.append("legacy")
            elif domain_name == "business" and idx < 86:
                pools.append("legacy")

            m["pools"] = pools

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)
        print(f"Migrated {domain_name}.yaml ({len(metrics)} metrics)")


if __name__ == "__main__":
    migrate()
