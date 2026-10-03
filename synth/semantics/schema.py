"""
synth/semantics/schema.py

Pydantic v2 validation schema for declarative YAML domain manifests (SEM-08).
Used strictly at build/test time to validate manifests under synth/semantics/domains/.
Enforces extra="forbid", chart types ⊂ ALL_CHART_TYPES, domain_id ∈ domain_tags,
and unique metric labels per manifest.
"""
import re
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from synth.semantics.enums import AxisRole, ScaleType

IDENTIFIER_REGEX = re.compile(r"^[a-z0-9_]+$")
ALL_CHART_TYPES = {"bar", "line", "scatter", "box", "area", "histogram", "pie", "heatmap"}


class MetricDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    scale_type: ScaleType
    role: AxisRole
    concept_stem: str
    unit: Optional[str] = None
    concept_group: Optional[str] = None
    domain_tags: List[str] = Field(default_factory=list)
    pools: Optional[List[str]] = Field(default_factory=list)

    @field_validator("label")
    @classmethod
    def validate_label_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Metric label cannot be empty")
        return s

    @field_validator("concept_stem")
    @classmethod
    def validate_concept_stem(cls, v: str) -> str:
        s = v.strip()
        if not s or not IDENTIFIER_REGEX.match(s):
            raise ValueError(f"concept_stem '{v}' must match pattern ^[a-z0-9_]+$ and not be empty")
        return s

    @field_validator("concept_group")
    @classmethod
    def validate_concept_group(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        if not s:
            return None
        if not IDENTIFIER_REGEX.match(s):
            raise ValueError(f"concept_group '{v}' must match pattern ^[a-z0-9_]+$")
        return s


class TitleDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    allowed_chart_types: List[str] = Field(default_factory=list)
    domain_tags: List[str] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def validate_title_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Title cannot be empty")
        return s

    @field_validator("allowed_chart_types")
    @classmethod
    def validate_allowed_chart_types(cls, v: List[str]) -> List[str]:
        for ctype in v:
            if ctype not in ALL_CHART_TYPES:
                raise ValueError(f"Invalid chart type '{ctype}'. Allowed chart types: {sorted(ALL_CHART_TYPES)}")
        return v


class PairDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    control: str
    treatment: str
    domain_category: str
    context_tags: List[str] = Field(default_factory=list)

    @field_validator("control", "treatment", "domain_category")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Pair control, treatment, and domain_category cannot be empty")
        return s

    @model_validator(mode="after")
    def validate_distinct_control_treatment(self) -> "PairDefinition":
        if self.control == self.treatment:
            raise ValueError(f"Comparative pair control and treatment must be distinct (got '{self.control}')")
        return self


class DomainManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_version: Optional[int] = None
    order: Optional[int] = None
    domain_id: str
    display_name: str
    is_scientific: Optional[bool] = None
    default_weight: float = 1.0
    default_fallbacks: Optional[List[str]] = None
    capabilities: Optional[Dict[str, bool]] = None
    metrics: List[MetricDefinition] = Field(default_factory=list)
    titles: List[TitleDefinition] = Field(default_factory=list)
    comparative_pairs: List[PairDefinition] = Field(default_factory=list)

    @field_validator("domain_id")
    @classmethod
    def validate_domain_id(cls, v: str) -> str:
        s = v.strip()
        if not s or not IDENTIFIER_REGEX.match(s):
            raise ValueError(f"domain_id '{v}' must match pattern ^[a-z0-9_]+$ and not be empty")
        return s

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("display_name cannot be empty")
        return s

    @model_validator(mode="after")
    def validate_domain_invariants(self) -> "DomainManifest":
        seen_labels = set()
        for idx, m in enumerate(self.metrics):
            if m.label in seen_labels:
                raise ValueError(f"Duplicate metric label in manifest: '{m.label}' (entry index {idx})")
            seen_labels.add(m.label)

        LEGACY_METRIC_TAG_EXCEPTIONS = {
            "biomedical": {"Pore Volume (cm³/g)", "Voltage (mV)"},
        }

        if self.manifest_version is not None and self.manifest_version >= 2 and self.domain_id != "common":
            exceptions = LEGACY_METRIC_TAG_EXCEPTIONS.get(self.domain_id, set())
            for idx, m in enumerate(self.metrics):
                if m.label not in exceptions and self.domain_id not in m.domain_tags:
                    raise ValueError(
                        f"Metric '{m.label}' at index {idx} in {self.domain_id}.yaml must include domain_id in domain_tags"
                    )
            if self.domain_id != "demographic":
                for idx, t in enumerate(self.titles):
                    if self.domain_id not in t.domain_tags:
                        raise ValueError(
                            f"Title '{t.title}' at index {idx} in {self.domain_id}.yaml must include domain_id in domain_tags"
                        )

        if self.default_fallbacks is not None:
            if len(self.default_fallbacks) != 2:
                raise ValueError(
                    f"default_fallbacks must contain exactly 2 items [x_fallback, y_fallback], got {len(self.default_fallbacks)}"
                )

        return self
