import httpx
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, redirect, render

from src.config import settings as api_settings

from .models import QueryHistory

API_BASE_URL = f"http://{api_settings.API_HOST}:{api_settings.API_PORT}/api/v1"

@login_required
def home_view(request):
    context = {}
    if request.method == 'POST':
        query_text = request.POST.get('query')
        if query_text:
            try:
                headers = {"X-User-Id": str(request.user.id)}
                with httpx.Client(base_url=API_BASE_URL, timeout=120.0, headers=headers) as client:
                    api_request_data = {"query": query_text}
                    response = client.post("/query", json=api_request_data)
                    response.raise_for_status()
                    
                    api_data = response.json()
                    
                    context['result'] = api_data
                    context['query'] = query_text

            except httpx.HTTPStatusError as e:
                context['error'] = f"API Error: {e.response.status_code} - {e.response.text}"
            except Exception as e:
                context['error'] = f"An unexpected error occurred: {e}"

    return render(request, 'history/home.html', context)

@login_required
def history_list_view(request):
    history_qs = QueryHistory.objects.filter(user=request.user).order_by('-created_at')
    
    paginator = Paginator(history_qs, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'history/history_list.html', {'page_obj': page_obj})

@login_required
def history_detail_view(request, pk):
    query_instance = get_object_or_404(QueryHistory, pk=pk, user=request.user)
    return render(request, 'history/history_detail.html', {'query': query_instance})

@login_required
def history_delete_view(request, pk):
    query_instance = get_object_or_404(QueryHistory, pk=pk, user=request.user)
    if request.method == 'POST':
        query_instance.delete()
        return redirect('history_list')
    
    return render(request, 'history/history_confirm_delete.html', {'query': query_instance})
