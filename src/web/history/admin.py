from django.contrib import admin

from .models import QueryHistory


@admin.register(QueryHistory)
class QueryHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'query_text', 'created_at', 'confidence_score', 'llm_model')
    list_filter = ('user', 'created_at', 'llm_provider')
    search_fields = ('query_text', 'response_md')
    readonly_fields = [field.name for field in QueryHistory._meta.fields]

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
