import datetime
from sqlalchemy import (
    Column,
    BigInteger,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    JSON,
    LargeBinary,
    Float
)
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    file_path = Column(Text, nullable=False, unique=True)
    source_url = Column(Text)
    title = Column(Text)
    domain = Column(Text, index=True)
    content_hash = Column(LargeBinary(32), nullable=False, index=True, unique=True) # blake2b-256
    full_text = Column(Text, nullable=False)
    meta_data = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, path='{self.file_path}')>"

class Chunk(Base):
    __tablename__ = 'chunks'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    document_id = Column(BigInteger, ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    embedding = Column(Vector(1024), nullable=False)
    # tsvector будет управляться триггером в БД, в модели его можно не объявлять
    meta_data = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint('document_id', 'chunk_index', name='_document_chunk_uc'),
    )

    def __repr__(self):
        return f"<Chunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"

class QueryHistory(Base):
    __tablename__ = 'query_history'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False, index=True) # Предполагается, что ID из Django auth_user
    query_text = Column(Text, nullable=False)
    query_embedding = Column(Vector(1024), nullable=False)
    response_md = Column(Text, nullable=False)
    sources_json = Column(JSON, nullable=False)
    llm_provider = Column(String, nullable=False)
    llm_model = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<QueryHistory(id={self.id}, user_id={self.user_id})>"