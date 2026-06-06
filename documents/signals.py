from django.db.models.signals import post_save
from django.dispatch import receiver
from documents.models import Document
from documents.processing import process_document


@receiver(post_save, sender=Document)
def process_document_on_save(sender, instance, created, **kwargs):
    """
    Automatically process a document after it's saved.
    Args:
        sender:the model class (Document)
        instance: the specific Document object that was saved
        created: True if this is a new record, False if it's an update
        **kwargs: other signal data we don't need
    """
    if created:
        print(f"Signal received: new document '{instance.title}'")
        process_document(instance)