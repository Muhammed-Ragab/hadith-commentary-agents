"""Pydantic schemas for Sahih Muslim transmission clusters."""
from pydantic import BaseModel, Field, model_validator


class Transmission(BaseModel):
    ref: str = Field(min_length=1)
    isnad: str = Field(min_length=1)
    matn: str = ""
    references_sibling: bool = False

    @model_validator(mode="after")
    def require_matn_or_sibling(self):
        if not self.matn.strip() and not self.references_sibling:
            raise ValueError("transmission requires matn or a sibling reference")
        return self


class Cluster(BaseModel):
    cluster_ref: str = Field(min_length=1)
    # The canonical Abd al-Baqi number the source prints as "(1046)" at the head of an isnad.
    # This is what a paper cites, but only 74% of records carry one and parallel routes share
    # it, so cluster_ref stays the unique key and this rides alongside it.
    muslim_number: str = ""
    kitab: str = Field(min_length=1)
    bab: str = Field(min_length=0)
    transmissions: list[Transmission] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_transmission_refs(self):
        refs = [transmission.ref for transmission in self.transmissions]
        if len(refs) != len(set(refs)):
            raise ValueError("transmission refs must be unique within a cluster")
        return self
