import json
from typing import Any, Dict, List, Tuple, cast

from sqlalchemy.orm import Session

from src.db.models import QueryHistory


class QueryHistoryService:
    """
    Сервисный слой для работы с историей запросов.
    Инкапсулирует логику сохранения и retrieval исторических данных.
    """

    @staticmethod
    def save(
        db: Session,
        user_id: int,
        query_text: str,
        query_embedding: list,
        result: Dict[str, Any],
        llm_client,
    ) -> int:
        """
        Сохраняет результат запроса в базу данных.

        Args:
            db: Сессия SQLAlchemy
            user_id: ID пользователя
            query_text: Текст запроса
            query_embedding: Эмбеддинг запроса (list[float])
            result: Результат от RAG-движка (словарь с полями response_md, sources, confidence_score и т.д.)
            llm_client: Объект LLM-клиента (имеет атрибуты provider и model)

        Returns:
            ID созданной записи в истории
        """
        history_entry = QueryHistory(
            user_id=user_id,
            query_text=query_text,
            query_embedding=query_embedding,
            response_md=result["response_md"],
            sources_json=json.loads(
                json.dumps([s for s in result.get("sources", [])], default=str)
            ),
            llm_provider=llm_client.provider,
            llm_model=llm_client.model,
            confidence_score=result.get("confidence_score", 0.0),
        )
        db.add(history_entry)
        db.commit()
        db.refresh(history_entry)
        return cast(int, history_entry.id)

    @staticmethod
    def get_user_history(
        db: Session, user_id: int, page: int = 1, limit: int = 10
    ) -> Tuple[List[QueryHistory], int]:
        """
        Получает историю запросов пользователя с пагинацией.

        Args:
            db: Сессия SQLAlchemy
            user_id: ID пользователя
            page: Номер страницы (начиная с 1)
            limit: Количество записей на странице

        Returns:
            Кортеж (список записей, общее количество записей)
        """
        query = (
            db.query(QueryHistory)
            .filter(QueryHistory.user_id == user_id)
            .order_by(QueryHistory.created_at.desc())
        )
        total = query.count()
        offset = (page - 1) * limit
        items = query.offset(offset).limit(limit).all()
        return items, total
