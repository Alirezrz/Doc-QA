from django.db import models
from documents.models import Document

class QAHistory(models.Model):
    """
    Stores every question asked and its answer.
    """
    question = models.TextField()
    answer = models.TextField()

    sources = models.ManyToManyField(Document, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.question[:80]

    class Meta:
        verbose_name_plural = "QA History"
        ordering = ['-created_at']