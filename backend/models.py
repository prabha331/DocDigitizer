import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship

from backend.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50))          # invoice / prescription / lab_report / handwritten_notes
    language = Column(String(10))           # en / hi / ta
    raw_text = Column(Text)
    extracted_json = Column(JSON)
    status = Column(String(20), default="pending")  # pending / completed / failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    fields = relationship(
        "ExtractedField",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    chat_history = relationship(
        "ChatHistory",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    def __repr__(self):
        return f"<Document id={self.id} filename={self.filename!r} status={self.status!r}>"


class ExtractedField(Base):
    __tablename__ = "extracted_fields"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    field_name = Column(String(100))
    field_value = Column(Text)
    confidence_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="fields")

    def __repr__(self):
        return f"<ExtractedField doc_id={self.doc_id} field={self.field_name!r} value={self.field_value!r}>"


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20))   # user / assistant
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chat_history")

    def __repr__(self):
        return f"<ChatHistory doc_id={self.doc_id} role={self.role!r} message={self.message[:40]!r}>"
