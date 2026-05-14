"""
RAG_Testing.py

Converted from RAG_Testing.ipynb.

Implementing Vector Search
==========================

Vector search finds similar items by comparing numerical representations
(embeddings) of data.

Core Concepts
-------------
    Text/Image -> Embedding Model -> Vector [0.2, 0.8, 0.1, ...] -> Store & Search

Cosine Similarity = (A . B) / (||A|| * ||B||)

Similarity metrics:
- Cosine similarity - angle between vectors (most common)
- Euclidean distance - straight-line distance
- Dot product       - magnitude + direction
"""

# =============================================================================
# Cell 1: Simple Vector Search (from scratch) using sentence-transformers + NumPy.
# -----------------------------------------------------------------------------
# This cell demonstrates the bare-bones mechanics of a semantic search engine:
#   1. Encode each document into a fixed-length embedding vector.
#   2. Encode the user query into the same embedding space.
#   3. Score every document against the query with cosine similarity.
#   4. Return the top-k highest-scoring documents.
# It is intentionally not optimized (no FAISS, no batching, no persistence)
# so the math is easy to follow.
# =============================================================================

import numpy as np
from sentence_transformers import SentenceTransformer


class SimpleVectorSearch:
    """Minimal in-memory vector search over a list of short documents."""

    def __init__(self):
        # 'all-MiniLM-L6-v2' is a lightweight 384-dim sentence embedding model
        # that's fast on CPU and good enough for demos / small corpora.
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectors = []     # parallel list of embedding vectors
        self.documents = []   # parallel list of original document strings

    def add_documents(self, docs: list[str]):
        """Encode each document into a vector and remember both."""
        self.documents.extend(docs)
        # .encode() returns a NumPy array shaped (n_docs, embedding_dim).
        embeddings = self.model.encode(docs)
        self.vectors.extend(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k documents most similar to `query`."""
        # Encode the query in the SAME embedding space as the documents.
        query_vector = self.model.encode([query])[0]

        # Stack the stored doc vectors into a matrix of shape (n_docs, dim)
        # so we can compute all similarities in one vectorized op.
        vectors_matrix = np.array(self.vectors)

        # Cosine similarity = (A . B) / (||A|| * ||B||)
        # Numerator: dot product of every doc vector with the query vector.
        # Denominator: product of L2 norms (per-doc norm * query norm).
        similarities = np.dot(vectors_matrix, query_vector) / (
            np.linalg.norm(vectors_matrix, axis=1) * np.linalg.norm(query_vector)
        )

        # argsort gives ascending order; [::-1] reverses to descending;
        # [:top_k] keeps only the best matches.
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return a list of {document, score} dicts, ordered by relevance.
        return [
            {"document": self.documents[i], "score": float(similarities[i])}
            for i in top_indices
        ]


def run_simple_vector_search_demo():
    """Quick sanity check: index four short sentences and run three queries."""
    search = SimpleVectorSearch()
    search.add_documents([
        "Python is a programming language",
        "Machine learning uses neural networks",
        "Vector search finds similar content",
        "Dogs are popular pets",
    ])

    # Query 1: AI/ML topic — expect the "machine learning" sentence on top.
    results = search.search("How does AI work?")
    for r in results:
        print(f"{r['score']:.3f} | {r['document']}")

    print('##########################################')

    # Query 2: Pet/animal topic — expect the "dogs" sentence on top.
    results = search.search("Where can I find puppy?")
    for r in results:
        print(f"{r['score']:.3f} | {r['document']}")

    print('##########################################')

    # Query 3: Cricket — no document is really about cricket, so scores
    # should all be relatively low (a useful negative example).
    results = search.search("Is the game of cricket the most popular?")
    for r in results:
        print(f"{r['score']:.3f} | {r['document']}")


# =============================================================================
# Cell 2: Utility — wipe the main Chroma collection used below.
# Uncomment and run only when you want to rebuild the index from the PDF.
# vector_db.delete_collection()
# =============================================================================


# =============================================================================
# Cell 3: Build a persistent Chroma vector store from a PDF using HuggingFace
# embeddings.
# -----------------------------------------------------------------------------
# Pipeline:
#   PDF -> page-level Documents -> ~1000-char chunks -> 768-dim embeddings
#       -> persisted Chroma collection on disk.
# We use the higher-quality `all-mpnet-base-v2` model (768 dims) rather than
# the smaller MiniLM (384 dims) because retrieval quality matters more here
# than speed.
# =============================================================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def build_vector_store(
    pdf_path: str = "/Users/rajeevkumar/Downloads/Employee Handbook - US.pdf",
    persist_directory: str = "./chroma_mpnet_db",
    collection_name: str = "employee_handbook_mpnet",
):
    """Load a PDF, chunk it, embed the chunks, and persist them in Chroma."""
    # -------------------------------------------------------------------------
    # 1. Configure the embedding model.
    #    - all-mpnet-base-v2 produces 768-dim vectors with strong semantic quality.
    #    - MiniLM is left commented as a faster, lower-quality alternative.
    #    - device='cpu' is portable; switch to 'cuda' if a GPU is available.
    #    - normalize_embeddings=False keeps raw vectors; Chroma will use cosine
    #      distance internally, so explicit normalization isn't required here.
    # -------------------------------------------------------------------------
    model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}            # Use 'cuda' if an NVIDIA GPU is available
    encode_kwargs = {'normalize_embeddings': False}

    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # -------------------------------------------------------------------------
    # 2. Load and chunk the source PDF.
    #    PyPDFLoader returns one Document per page; the splitter then breaks
    #    those pages into overlapping ~1000-char windows so a single chunk
    #    fits comfortably in the embedding model's context and retrieval is
    #    fine-grained. chunk_overlap=150 keeps continuity across chunk
    #    boundaries so a sentence that straddles two chunks is still
    #    retrievable.
    # -------------------------------------------------------------------------
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)

    # -------------------------------------------------------------------------
    # 3. Embed every chunk and persist them in a local Chroma collection.
    #    `persist_directory` makes the index survive across notebook restarts
    #    so we don't have to re-embed on every run.
    # -------------------------------------------------------------------------
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=hf_embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print(f"Finished loading {len(docs)} chunks using all-mpnet-base-v2.")
    return vector_db, hf_embeddings


# =============================================================================
# Cell 4: Inspect what Chroma actually stored — original text + embedding
# vectors. Useful for sanity-checking chunk size, page boundaries, and vector
# shape.
# =============================================================================

import chromadb


def inspect_collection(
    persist_directory: str = "./chroma_mpnet_db",
    collection_name: str = "employee_handbook_mpnet",
    start: int = 60,
    end: int = 62,
):
    """Print a sample of stored documents and their embedding vectors."""
    # 1. Initialize the Chroma Client and connect to the collection.
    client = chromadb.PersistentClient(path=persist_directory)

    # 2. Get a list of all collection objects.
    collections = client.list_collections()
    print(f"Collections in Chroma DB: {[col.name for col in collections]}")

    # 3. Access the specific collection where we stored the chunks and embeddings.
    collection = client.get_collection(name=collection_name)

    # 4. Pull EVERYTHING out of the collection. By default Chroma returns ids
    # only; we have to explicitly ask for documents / metadata / embeddings.
    results = collection.get(
        include=["documents", "metadatas", "embeddings"],
    )

    print(f"Total Documents Chunks Created : {len(results['ids'])}")

    # Look at a small window of chunks. Adjust start/end to inspect other regions.
    # We iterate by index (not by dict key) so we can line up
    # documents <-> embeddings <-> metadatas by position.
    for i in range(start, end):
        print(f"\n--- Index {i} ---")

        # Chroma can return None for fields that weren't stored; the explicit
        # `is not None` check avoids accidentally truth-testing a NumPy array
        # (which raises ValueError on ambiguous truthiness).
        if results['documents'] is not None:
            print(f"Original Document: {results['documents'][i]}")

        if results['embeddings'] is not None:
            # Wrap in np.array so we can call .shape regardless of whether
            # Chroma handed us a list or already a NumPy array.
            val = np.array(results['embeddings'][i])
            print(f"Rows and Dimensions : {val.shape}")          # expect (768,)
            print(f"Embedded Vector (First 10) : {val[:10]}")    # peek at the values

    return results


# =============================================================================
# Cell 5: Scratch — uncomment to dump the full `results` dict.
# Warning: with ~1k chunks this prints a *lot*.
# print(results)
# =============================================================================


# =============================================================================
# Cell 6: Plain similarity search (no LLM yet) — ask the vector DB for the
# top-k chunks closest to a natural-language query. This is the "Retrieval"
# half of RAG.
# =============================================================================

def similarity_search_demo(
    hf_embeddings,
    query: str = "Tell me about paternity policy?",
    persist_directory: str = "./chroma_mpnet_db",
    collection_name: str = "employee_handbook_mpnet",
    k: int = 5,
):
    """Run a similarity search against the persisted Chroma collection."""
    vector_db_read = Chroma(
        embedding_function=hf_embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    # k=5 -> return the five most relevant chunks. Larger k = more context for
    # the downstream LLM but more noise and more tokens.
    results = vector_db_read.similarity_search(query, k=k)

    # Print each retrieved chunk on its own block so we can eyeball whether
    # the retriever is finding the right section of the handbook.
    for doc in results:
        print(f"Result: {doc.page_content}\n")

    return results


# =============================================================================
# Cell 7: Second retrieval probe — same mechanism, different topic.
# Comparing the two queries side-by-side is a quick way to gauge whether the
# embedding model can distinguish unrelated policy sections.
# =============================================================================

def similarity_search_401k(hf_embeddings):
    """Probe the index with a 401-k query."""
    return similarity_search_demo(
        hf_embeddings,
        query="Tell me about 401-k policy?",
        k=5,
    )


# =============================================================================
# Cell 8: Baseline — call Claude directly with no retrieved context.
# This is the "no-RAG" control — it shows what the model can answer from its
# own pretraining, with no access to the AIG handbook.
# =============================================================================

import anthropic
import os
from dotenv import load_dotenv


def claude_baseline(question: str = "How do I implement a vector search?"):
    """Send a question to Claude with no retrieved context."""
    # Pull ANTHROPIC_API_KEY out of a local .env file into os.environ.
    load_dotenv()

    # 1. Initialize the Anthropic client.
    #    If ANTHROPIC_API_KEY is already in the environment, calling
    #    anthropic.Anthropic() with no args would also work — passing it
    #    explicitly here makes the dependency obvious.
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    # 2. Send a single user turn to the model.
    #    max_tokens caps the response size; the model stops earlier if it
    #    naturally finishes.
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {"role": "user", "content": question}
        ],
    )

    # 3. The API returns a list of content blocks; for a simple text reply
    #    the first block is the text we want.
    print(message.content[0].text)
    return message


# =============================================================================
# Cell 9: Full RAG (Retrieval-Augmented Generation) pipeline.
#   1. Use the vector DB to fetch chunks relevant to the user's question.
#   2. Concatenate those chunks into a context block.
#   3. Send {system prompt + context + question} to Claude.
#   4. Print Claude's grounded answer.
# This is the pattern that lets the model answer questions about a private
# document (the AIG Employee Handbook) it was never trained on.
# =============================================================================

def rag_query(
    vector_db,
    query: str = "Tell me about work attire policy?",
    k: int = 10,
):
    """Run the full RAG pipeline for a given query."""
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # -------------------------------------------------------------------------
    # 1. Retrieval — pull the top-k most relevant chunks from Chroma.
    #    Alternate queries kept as quick A/B test toggles:
    # query = "Tell me about paternity policy?"
    # query = "Tell me about leaves entitlement?"
    # query = "Tell me about unapproved leave policy and its implications of employment?"
    # query = "Tell me about work attire policy?"
    # -------------------------------------------------------------------------
    # k=10 gives the LLM a fairly wide context window over the handbook.
    # Tune k up for recall-heavy questions, down to reduce token cost / noise.
    docs = vector_db.similarity_search(query, k=k)

    # -------------------------------------------------------------------------
    # 2. Stitch retrieved chunks together into a single context string.
    #    A blank line between chunks helps the model treat them as separate passages.
    # -------------------------------------------------------------------------
    context = "\n\n".join([doc.page_content for doc in docs])

    # -------------------------------------------------------------------------
    # 3. Build the prompt.
    #    - The system prompt sets the role and tells the model to ground answers
    #      in the provided context, and to admit when the answer isn't present.
    #    - The user message bundles the retrieved context with the question.
    # -------------------------------------------------------------------------
    system_prompt = """You are an expert data assistant. Use the provided context
to answer the user's question. If the answer isn't in the context, say so."""

    user_message = f"""Context:
{context}

Question: {query}"""

    # -------------------------------------------------------------------------
    # 4. Generation — send the grounded prompt to Claude and print the answer.
    # -------------------------------------------------------------------------
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    print(response.content[0].text)
    return response


# =============================================================================
# Main entry point — runs the full notebook flow end-to-end.
# Comment out any step you don't need.
# =============================================================================

if __name__ == "__main__":
    # 1. From-scratch vector search demo.
    run_simple_vector_search_demo()

    # 2. Build / refresh the Chroma collection from the PDF.
    vector_db, hf_embeddings = build_vector_store()

    # 3. Peek at the stored chunks + their embedding vectors.
    inspect_collection()

    # 4. Run a couple of retrieval-only probes against the index.
    similarity_search_demo(hf_embeddings, query="Tell me about paternity policy?")
    similarity_search_401k(hf_embeddings)

    # 5. Baseline Claude call (no RAG).
    claude_baseline("How do I implement a vector search?")

    # 6. Full RAG pipeline grounded in the handbook.
    rag_query(vector_db, query="Tell me about work attire policy?")
