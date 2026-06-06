"""
This file handles all document processing:
1. Extract text from .docx files
2. Split text into chunks
3. Generate embeddings for each chunk
"""

from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    """Load embedding model once and reuse it."""
    global _model
    if _model is None:
        print("Loading embedding model... (first time only)")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded!")
    return _model


def extract_text(file_path):
    """
    Extract all text from a .docx file.
    We join them all with newlines to get the full text.
    Returns All text as one big string
    """
    doc = DocxDocument(file_path)

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    return '\n'.join(paragraphs)


def split_into_chunks(text: str, chunk_size: int = 80, overlap: int = 15):

    words = text.split()  
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])  
        chunks.append(chunk)
        start += chunk_size - overlap      

    return chunks


def generate_embedding(text: str) -> list:
    """
    Convert text into a vector (list of 384 numbers).
    Similar texts → similar vectors → close together in vector space.
    """
    model = get_model()
    vector = model.encode(text)
    return vector.tolist()      # convert numpy array to plain Python list


def process_document(document):
    """
    extract → chunk → embed → save to DB.
    """
    from documents.models import DocumentChunk

    print(f"Processing: {document.title}")

    text = extract_text(document.file.path)
    document.extracted_text = text
    document.save()
    print(f"  Extracted {len(text)} characters")

    document.chunks.all().delete()

    chunks = split_into_chunks(text)
    print(f"  Split into {len(chunks)} chunks")

    # Step 4: For each chunk, generate embedding and save to DB
    for index, chunk_text in enumerate(chunks):
        embedding = generate_embedding(chunk_text)  # we are turning chunks into vectors here 

        chunk = DocumentChunk(
            document=document,
            content=chunk_text,
            chunk_index=index,
        )
        chunk.set_embedding(embedding)
        chunk.save()

    print(f"  Done! Saved {len(chunks)} chunks with embeddings")