"""
The RAG Pipeline — heart of the system.
"""
import time
import os
import numpy as np
from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from documents.processing import generate_embedding


def cosine_similarity(vec_a: list, vec_b: list):
    """
      1.0 = identical meaning
      0.0 = completely unrelated
     -1.0 = opposite meaning

    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    dot_product = np.dot(a, b)              # A · B
    magnitude_a = np.linalg.norm(a)        # |A|
    magnitude_b = np.linalg.norm(b)        # |B|

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0 

    return float(dot_product / (magnitude_a * magnitude_b))


def find_relevant_chunks(question: str, top_k: int = 4):
    """
    Find the top_k document chunks most relevant to the question.
    """
    from documents.models import DocumentChunk

    question_vector = generate_embedding(question)
    #if user specifies the document the part "select_related('document')" :
    chunks = list(
        DocumentChunk.objects.filter(
            embedding_json__isnull=False
        ).select_related('document')
    )

    if not chunks:
        return []

    scored = []
    for chunk in chunks:
        chunk_vector = chunk.get_embedding()
        if chunk_vector:
            score = cosine_similarity(question_vector, chunk_vector)
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]


def get_llm():
    """
    Create the LLM client pointing at OpenRouter.
    OpenRouter uses the same API format as OpenAI.
    """
    return ChatOpenAI(
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=settings.OPENROUTER_MODEL,
        temperature=0.1,    # 0 = deterministic/factual, 1 = creative/random
        max_tokens=1000,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "DocQA"
        }
    )


def ask_question(question: str):
    """
    takes a question, returns an answer.
    
    flow:
    1. Find relevant chunks (vector search)
    2. Format chunks as context text
    3. Build prompt: system message + context + question
    4. Send to LLM
    5. Return answer + source documents
    """
    start_time = time.time()
    print(f"\n[Pipeline] Question: {question}")

    chunks = find_relevant_chunks(question, top_k=4)
    print(f"[Pipeline] Found {len(chunks)} relevant chunks")

    if chunks:
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"[From: {chunk.document.title}, section {chunk.chunk_index}]\n"
                f"{chunk.content}"
            )
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "No relevant documents found."

    messages =[
        SystemMessage(content=(
            "You are a helpful assistant that answers questions "
            "based strictly on the provided document context. "
            "If the answer is not in the context, say so honestly. "
            "Do not make up information. Be concise and direct."
        )),
        HumanMessage(content=(
            f"Context from documents:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on the context above:"
        ))
    ]

    llm = get_llm()
    print(f"[Pipeline] Calling {settings.OPENROUTER_MODEL}...")
    response = llm.invoke(messages)
    answer = response.content
    print(f"[Pipeline] Got answer ({len(answer)} chars)")

    source_docs = list({chunk.document for chunk in chunks})

    return {
        'answer': answer,
        'source_documents': source_docs,
        'chunks_used': len(chunks),
        'response_time_seconds': round(time.time() - start_time, 2),
        'model_used': settings.OPENROUTER_MODEL,
    }