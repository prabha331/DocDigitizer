from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel


class ExtractResponse(BaseModel):
    doc_id: UUID
    filename: str
    doc_type: str
    language: str
    extracted_json: dict[str, Any]
    raw_text: str
    status: str
    all_lines: list[str] = []

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    doc_id: UUID
    message: str
    language: str = "en"


class ChatResponse(BaseModel):
    doc_id: UUID
    response: str
    role: str = "assistant"


class ExtractedFieldResponse(BaseModel):
    id: UUID
    doc_id: UUID
    field_name: Optional[str]
    field_value: Optional[str]
    confidence_score: float
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    id: UUID
    doc_id: UUID
    role: str
    message: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    id: UUID
    filename: str
    file_type: Optional[str]
    language: Optional[str]
    raw_text: Optional[str]
    extracted_json: Optional[dict[str, Any]]
    status: str
    created_at: datetime
    updated_at: datetime
    fields: list[ExtractedFieldResponse] = []

    model_config = {"from_attributes": True}
