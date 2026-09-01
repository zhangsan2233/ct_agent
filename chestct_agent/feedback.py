"""Versioned, reviewable human feedback contracts for safe model improvement."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


LabelStatus = Literal["positive", "negative", "uncertain"]
FeedbackStatus = Literal["pending", "approved", "rejected", "used_for_calibration", "used_for_training"]


class ImageAnnotation(BaseModel):
    slice_index: int = Field(ge=0)
    image_path: str = Field(default="", max_length=1000)
    bbox_2d: list[int] = Field(min_length=4, max_length=4)
    note: str = Field(default="", max_length=500)


class FeedbackItem(BaseModel):
    label: str = Field(min_length=1, max_length=128)
    corrected_status: LabelStatus
    reason: str = Field(default="", max_length=1000)
    annotations: list[ImageAnnotation] = Field(default_factory=list, max_length=20)


class FeedbackSubmission(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    reviewer: str = Field(min_length=1, max_length=128)
    reviewer_role: Literal["user", "clinician", "administrator"] = "user"
    model_version: str = Field(default="unknown", max_length=128)
    items: list[FeedbackItem] = Field(min_length=1)


class FeedbackReview(BaseModel):
    status: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1, max_length=128)
    note: str = Field(default="", max_length=1000)
