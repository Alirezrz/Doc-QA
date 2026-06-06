"""
Serializers = translators between Python objects and JSON.
"""
from rest_framework import serializers
from documents.models import Document

class DocumentSerializer(serializers.ModelSerializer):
    chunk_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Document       
        fields = [             
            'id',
            'title',
            'file',
            'extracted_text',
            'uploaded_at',
            'updated_at',
            'chunk_count',
        ]
        read_only_fields = ['extracted_text', 'uploaded_at', 'updated_at']