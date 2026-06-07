from django import forms
from django.contrib import admin, messages
from django.shortcuts import render, redirect
from django.urls import path
from django.utils.html import format_html
from .models import QAHistory
from .pipeline import ask_question


class AskQuestionForm(forms.Form):
    question = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 4,
            'cols': 80,
            'placeholder': 'Type your question here...',
            'style': 'width:100%; font-size:15px; padding:10px; border-radius:6px; border:1px solid #ccc;'
        }),
        label='Your Question',
    )



@admin.register(QAHistory)
class QAHistoryAdmin(admin.ModelAdmin):
    list_display = ['short_question', 'short_answer', 'get_sources', 'created_at']
    readonly_fields = ['question', 'answer', 'get_sources', 'created_at']

    change_list_template = 'admin/qa/qahistory/change_list.html'

    def short_question(self, obj):
        return obj.question[:80] + ('...' if len(obj.question) > 80 else '')
    short_question.short_description = 'Question'

    def short_answer(self, obj):
        return obj.answer[:100] + ('...' if len(obj.answer) > 100 else '')
    short_answer.short_description = 'Answer'
    def get_sources(self, obj):
        return ', '.join([doc.title for doc in obj.sources.all()])
    get_sources.short_description = 'Sources'

    def get_urls(self):
        urls = super().get_urls()
        custom = [
            path('ask/', self.admin_site.admin_view(self.ask_view), name='qa_ask'),
        ]
        return custom + urls

    def ask_view(self, request):
        result = None
        form   = AskQuestionForm()

        if request.method == 'POST':
            form = AskQuestionForm(request.POST)
            if form.is_valid():
                question = form.cleaned_data['question'].strip()
                try:
                    raw = ask_question(question)

                    qa = QAHistory.objects.create(
                        question=question,
                        answer=raw['answer'],
                    )
                    qa.sources.set(raw['source_documents'])

                    result = {
                        'answer':  raw['answer'],
                        'sources': [doc.title for doc in raw['source_documents']],
                    }
                    messages.success(request, 'Answer generated and saved to history.')

                except Exception as e:
                    messages.error(request, f'Error: {str(e)}')

        context = {
            **self.admin_site.each_context(request),
            'title':  'Ask a Question',
            'form':   form,
            'result': result,
        }
        return render(request, 'admin/qa/qahistory/ask.html', context)