# Domain & Generation Console User Guide and Architecture Specification

This document provides complete reference documentation for the Domain & Generation Console (`scripts/manage_domains.py`), its headless administration engine (`synth.semantics.admin`), configuration architecture, spreadsheet ingestion contract, and disaster recovery procedures.

---

## 1. Architecture Overview & Configuration Precedence

The console features a headless core (`synth.semantics.admin`) driven either via the interactive `rich` Terminal User Interface (TUI) or headless CLI subcommands, operating over declarative YAML manifests in `synth/semantics/domains/`.

### 1.1 Layered Configuration Resolution

Settings follow a deterministic, layered hierarchy resolved by `config_loader.py` into an immutable `ResolvedConfig`:

```
┌────────────────────────────────────────────────────────┐
│ 1. config_defaults.py (Base defaults & documented schema)│
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 2. custom_config.py (Managed override block)           │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 3. --config <path> (Explicit .py or .json overrides)   │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 4. Environment Variables (e.g. DEBUG_MODE)             │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ 5. CLI Arguments (--set key=value, --num, --output)    │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│ ResolvedConfig (Deep copy + Provenance metadata)       │
└────────────────────────────────────────────────────────┘
```

* **Managed Block in `custom_config.py`**: Changes made through the TUI or `manage_domains.py config set` are persisted strictly between sentinels `# >>> managed by manage_domains — do not edit >>>` and `# <<< managed by manage_domains <<<`. Leaf assignments only record differences from defaults.
* **Opt-Out**: Running with `--no-custom` bypasses `custom_config.py`, evaluating only base defaults and CLI arguments.
* **Worker Process Isolation**: When executing parallel chart generation, `ProcessPoolExecutor` uses a module-level initializer (`_init_worker`) to transmit the resolved configuration to spawned workers.

---

## 2. Spreadsheet & CSV Ingestion Contract

The ingestion engine (`synth.semantics.admin.ingest`) accepts `.xlsx` (via `openpyxl`, `data_only=True`) and `.csv` (BOM-tolerant, dialect-sniffing for `,`, `;`, `\t` with UTF-8 and cp1252 fallback). Caps: $\le 10\text{ MB}$, $\le 20{,}000\text{ rows}$.

### 2.1 Sheet Naming
Sheet names are matched case- and space-insensitively:
* **Metrics**: `metrics`, `metric`, `axismetrics`
* **Titles**: `titles`, `title`, `charttitles`
* **Comparative Pairs**: `pairs`, `pair`, `comparativepairs`

### 2.2 Metrics Sheet Contract

| Column Name | Synonyms | Type | Description |
|---|---|---|---|
| **Label** (Required) | `metric`, `metriclabel`, `name`, `axislabel` | `str` | Display label (e.g. `Plasma Concentration (ng/mL)`). Normalized via Unicode NFC, whitespace collapse, and $\mu$ unification. |
| **Scale Type** | `scaletype`, `scale`, `type`, `datatype` | `str` | Must be one of `ScaleType` enum values: `continuous` (default), `categorical`, `discrete_count`, `percentage`, `temporal`. <br>**Note on `ordinal`:** Input specifying `ordinal` is mapped to `categorical` with a validation warning. |
| **Axis Role** | `axisrole`, `role`, `axis` | `str` | `dependent` (default) or `independent`. |
| **Unit** | `units`, `measurementunit` | `str` | Unit string (e.g. `mg/L`, `USD`, `s`). |
| **Concept Stem** | `conceptstem`, `stem` | `str` | Unique concept key matching `^[a-z0-9_]+$`. Automatically generated via `make_stem(label)` if omitted. |
| **Concept Group** | `conceptgroup`, `group` | `str` | Semantic sub-grouping (e.g. `pharmacokinetics`, `telemetry`). |
| **Domain Tags** | `domaintags`, `tags`, `domains` | `str` (comma-separated) | Target domain identifiers. Must include the target `domain_id`. |
| **Pools** | `pool` | `str` (comma-separated) | Optional pool assignment tags: `legacy`, `canonical_independent`, `histogram_y`. |

### 2.3 Titles Sheet Contract

| Column Name | Synonyms | Type | Description |
|---|---|---|---|
| **Title** (Required) | `charttitle`, `name` | `str` | Chart title text. |
| **Allowed Chart Types** | `allowedcharttypes`, `charttypes`, `types`, `charts` | `str` (comma-separated) | Supported chart types (`bar`, `line`, `scatter`, `box`, `histogram`, `area`). Defaults to all Cartesian types if blank. |
| **Domain Tags** | `domaintags`, `tags`, `domains` | `str` (comma-separated) | Domain associations. Must include target `domain_id`. |

### 2.4 Comparative Pairs Sheet Contract

| Column Name | Synonyms | Type | Description |
|---|---|---|---|
| **Control** (Required) | `controlgroup`, `base`, `baseline` | `str` | Baseline or control condition (e.g. `Vehicle`, `Standard of Care`). |
| **Treatment** (Required) | `treatmentgroup`, `intervention`, `test` | `str` | Comparison or treatment condition. Must differ from control. |
| **Domain Category** | `domain`, `category` | `str` | Owning domain identifier. |
| **Context Tags** | `tags` | `str` (comma-separated) | Context tags for pair sampling. |

---

## 3. Collision Policies and Classification

During pre-flight planning (`synth.semantics.admin.plan`), candidate entries are classified against existing manifests:

### 3.1 Classification Statuses
* **`NEW`**: Entity label does not exist anywhere in the corpus. Appended to the manifest.
* **`DUPLICATE`**: Exact normalized label match in the target domain. Handled according to collision policy.
* **`SHARED`**: Exact normalized label match in a different domain. By default, adds target `domain_id` to the existing metric's `domain_tags` without duplicating YAML entries.
* **`UNIT-VARIANT`**: Same concept stem as an existing metric but different unit (e.g. `Temperature (°C)` vs `Temperature (K)`). Allowed as distinct metric.
* **`SIMILAR`**: Unit-stripped fuzzy score $\ge 90\%$ (via `rapidfuzz`). Flagged for curator review to catch typos.
* **`INVALID`**: Validation failure (empty label, empty stem, control equals treatment). Dropped unconditionally.

### 3.2 Collision Policies (`--on-collision`)
* **`skip`** *(default)*: Ignores duplicates within the same domain; cross-domain matches become `SHARED`.
* **`merge`**: Merges tags and allowed chart types into the existing entry.
* **`overwrite`**: Replaces metric definition in-place within the same domain, strictly preserving its list index and `pools`.
* **`abort`**: Aborts the operation if any duplicate or similar match is encountered.

---

## 4. Transactional Safety, Backups, and Disaster Recovery

All modifications to YAML manifests in `synth/semantics/domains/` execute within a transactional engine (`synth.semantics.admin.apply`) guaranteeing zero half-written state:

### 4.1 Commit Pipeline
1. **Directory File Locking**: Acquires `domains/.lock` with `O_EXCL` and PID tracking. Automatically recovers from stale locks after 10 minutes.
2. **Snapshot Creation**: Full backup of all `.yaml` manifests is copied to `domains/.backups/<UTC_TIMESTAMP>/`.
3. **Pydantic Validation**: All candidate manifests in memory are validated against `DomainManifest` (`extra="forbid"`).
4. **Registry Invariant Gate**: Compiles an in-memory `SemanticRegistry` to guarantee that Cartesian sampling pools (bar, line, scatter, box, area, histogram) are non-degenerate.
5. **Atomic Serialization**: Writes files using `ruamel.yaml` in round-trip mode (preserving formatting and comments) through a process-tagged `.tmp` file, flushes, syncs with `os.fsync`, and renames atomically with `os.replace`.
6. **Cache Rebuild**: Calls `rebuild_cache(force=True)` to write an updated binary cache with content fingerprints.
7. **Automatic Rollback**: Any error during validation, writing, or optional verification tests triggers an immediate rollback from the snapshot.

### 4.2 Disaster Recovery Procedure

If a manual edit or interrupted operation corrupts manifests, restore directly to a snapshot backup:

```bash
# 1. List available backup timestamps
python scripts/manage_domains.py restore --list

# 2. Restore to a specific timestamp
python scripts/manage_domains.py restore 20261002_153012_123456

# 3. Verify manifest health
python scripts/manage_domains.py audit --strict
```

---

## 5. CLI & Headless Subcommand Reference

All studio operations can be run headlessly for CI/CD automation:

```bash
# Audit corpus for collisions and degenerate pools
python scripts/manage_domains.py audit [--strict] [--json]

# Generate Excel starter template with dropdown validation
python scripts/manage_domains.py template -o domain_template.xlsx

# Create a new domain manifest skeleton
python scripts/manage_domains.py new-domain aerospace --display-name "Aerospace" --scientific

# Ingest and apply metrics spreadsheet
python scripts/manage_domains.py import data.xlsx -d aerospace --on-collision skip -y

# Revalidate manifests and force binary cache rebuild
python scripts/manage_domains.py rebuild

# Inspect or modify generator settings
python scripts/manage_domains.py config show
python scripts/manage_domains.py config set num_images=50 engine=matplotlib

# Launch chart generation with real-time JSONL progress
python scripts/manage_domains.py generate -n 100 -o output/test_run --format detection
```

---

## 6. Corpus Audit Baseline & Remediation Log

Running `python scripts/manage_domains.py audit --strict` confirms zero degenerate pools across all Cartesian chart types:

* **Degenerate Pools**: None (`0` across all Cartesian types and orientations).
* **Tracked Invariants**: 21 warnings for legacy cross-tagged titles/metrics from Phase 0 baseline migrations (e.g. demographic titles with biomedical/business tags). These remain tracked and supported for backwards compatibility.
