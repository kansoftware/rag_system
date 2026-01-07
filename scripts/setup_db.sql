-- Включаем расширение, если не включено
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Таблица документов
CREATE TABLE IF NOT EXISTS documents (
  id            BIGSERIAL PRIMARY KEY,
  file_path     TEXT NOT NULL UNIQUE,
  source_url    TEXT,
  title         TEXT,
  domain        TEXT,
  content_hash  BYTEA NOT NULL UNIQUE,
  full_text     TEXT NOT NULL,
  meta_data      JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents USING hash(content_hash);

-- 2. Таблица чанков
CREATE TABLE IF NOT EXISTS chunks (
  id           BIGSERIAL PRIMARY KEY,
  document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index  INT NOT NULL,
  chunk_text   TEXT NOT NULL,
  token_count  INT NOT NULL,
  embedding    vector(1024) NOT NULL,
  chunk_text_tsv tsvector, -- Для гибридного поиска
  meta_data     JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(document_id, chunk_index)
);

-- Индекс для векторного поиска
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Индекс для полнотекстового поиска
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING GIN(chunk_text_tsv);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id, chunk_index);

-- Триггер для автоматического обновления tsvector
CREATE OR REPLACE FUNCTION chunks_tsvector_update() RETURNS trigger AS $$
BEGIN
  NEW.chunk_text_tsv :=
      setweight(to_tsvector('english', COALESCE(NEW.meta_data->>'section_title', '')), 'A') ||
      setweight(to_tsvector('english', NEW.chunk_text), 'B');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsvector_update ON chunks;
CREATE TRIGGER tsvector_update
  BEFORE INSERT OR UPDATE ON chunks
  FOR EACH ROW EXECUTE FUNCTION chunks_tsvector_update();

-- 3. Таблица истории запросов
CREATE TABLE IF NOT EXISTS query_history (
  id                BIGSERIAL PRIMARY KEY,
  user_id           BIGINT NOT NULL, -- FK на auth_user.id
  query_text        TEXT NOT NULL,
  query_embedding   vector(1024) NOT NULL,
  response_md       TEXT NOT NULL,
  sources_json      JSONB NOT NULL,
  llm_provider      TEXT NOT NULL,
  llm_model         TEXT NOT NULL,
  confidence_score  REAL NOT NULL,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_qh_user_time ON query_history(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_qh_embedding_hnsw
  ON query_history USING hnsw (query_embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);