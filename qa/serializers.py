from rest_framework import serializers
from qa.models import QAHistory

class QAHistorySerializer(serializers.ModelSerializer):
    sources = serializers.StringRelatedField(many=True, read_only=True)

    class Meta:
        model = QAHistory
        fields = ['id', 'question', 'answer', 'sources', 'created_at']
        read_only_fields = ['answer', 'sources', 'created_at']