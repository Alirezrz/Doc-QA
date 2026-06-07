from django.contrib import admin
from .models import QAHistory

@admin.register(QAHistory)
class QAHistoryAdmin(admin.ModelAdmin):
    list_display = ['question', 'created_at']
    readonly_fields = ['question', 'answer', 'sources', 'created_at']