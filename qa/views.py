from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from qa.models import QAHistory
from qa.serializers import QAHistorySerializer
from qa.pipeline import ask_question

class AskView(APIView):
    """Handle POST /api/ask/ — the main Q&A endpoint."""

    def post(self, request):
        question = request.data.get('question', '').strip()

        if not question:
            return Response(
                {'error': 'question field is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            result = ask_question(question)

            qa = QAHistory.objects.create(
                question=question,
                answer=result['answer']
            )
            qa.sources.set(result['source_documents'])

            serializer = QAHistorySerializer(qa)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response(
                {'error': f'An error occurred: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HistoryView(APIView):
    """Handle GET /api/history/ — list all past Q&A."""

    def get(self, request):
        history = QAHistory.objects.all()
        serializer = QAHistorySerializer(history, many=True)
        return Response(serializer.data)