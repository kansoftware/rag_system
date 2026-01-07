from django.db import models
from django.contrib.auth.models import User
# pgvector.django не нужен, так как мы не используем Django ORM для записи
# from pgvector.django import VectorField

class QueryHistory(models.Model):
    """
    Представление таблицы query_history для Django ORM.
    `managed = False` означает, что Django не будет управлять этой таблицей.
    """
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, db_column='user_id')
    query_text = models.TextField()
    # query_embedding = VectorField(dimensions=1024) # Django не будет знать об этом типе
    response_md = models.TextField()
    sources_json = models.JSONField()
    llm_provider = models.CharField(max_length=255)
    llm_model = models.CharField(max_length=255)
    confidence_score = models.FloatField()
    created_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'query_history'
        verbose_name = "Query History"
        verbose_name_plural = "Query Histories"
        ordering = ['-created_at']

    def __str__(self):
        return f"Query by {self.user.username} at {self.created_at.strftime('%Y-%m-%d %H:%M')}"
