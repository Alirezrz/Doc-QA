"""
  GET    /api/documents/ -> list all documents
  POST   /api/documents/-> upload a new document
  GET    /api/documents/1/-> get document with id=1
  PUT    /api/documents/1/-> update document with id=1
  DELETE /api/documents/1/  -> delete document with id=1
"""

from rest_framework import viewsets
from documents.models import Document
from documents.serializers import DocumentSerializer

class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all().order_by('-uploaded_at')
    serializer_class = DocumentSerializer