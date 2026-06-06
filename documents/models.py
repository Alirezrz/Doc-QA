import json
from django.db import models

class Document(models.Model):
    """
    Represents one uploaded Word document.
    Database table: documents_document
    Columns: id, title, file, extracted_text, uploaded_at, updated_at
    """

    title = models.CharField(max_length=255)

    file = models.FileField(upload_to='documents/')

    extracted_text = models.TextField(blank=True)

    uploaded_at = models.DateTimeField(auto_now_add=True)

    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        # This controls how a Document appears in the admin panel
        return self.title

    @property
    def chunk_count(self):
        return self.chunks.count()


class DocumentChunk(models.Model):
    """
    One piece of a document after splitting.
    LLMs have a context limit — you can't send 50 pages to them.
    So we split documents into ~500 word pieces.
    We only send the RELEVANT pieces to the LLM, not the whole doc.
    
    Database table: documents_documentchunk
    """

    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name='chunks'
    )

    content = models.TextField()

    chunk_index = models.IntegerField(default=0)

    embedding_json = models.TextField(null=True, blank=True)

    def set_embedding(self, vector: list):
        """Convert a Python list to JSON string and save it."""
        self.embedding_json = json.dumps(vector)

    def get_embedding(self) -> list:
        """Convert the stored JSON string back to a Python list."""
        if self.embedding_json:
            return json.loads(self.embedding_json)
        return []

    def __str__(self):
        return f"{self.document.title} — Chunk {self.chunk_index}"