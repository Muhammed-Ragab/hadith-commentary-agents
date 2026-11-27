from typing import Literal

from pydantic import BaseModel, Field, model_validator


ClaimStatus = Literal["unchecked", "verified", "unverified", "unverified_model_memory"]


class Claim(BaseModel):
    text: str = Field(min_length=1)
    arabic_evidence: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    status: ClaimStatus = "unchecked"


class Finding(BaseModel):
    skill_id: str = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
    note: str = ""


class Verdict(BaseModel):
    kind: Literal["jam", "naskh", "tarjih", "tawaqquf"]
    reading: str = Field(min_length=1)
    supported_by: list[str] = Field(default_factory=list)
    unsettled: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_support_unless_tawaqquf(self):
        if self.kind != "tawaqquf" and not self.supported_by:
            raise ValueError("non-tawaqquf verdicts require support")
        return self


class Diagnosis(BaseModel):
    skill_plan: list[str] = Field(min_length=1)
    reason: str = ""
