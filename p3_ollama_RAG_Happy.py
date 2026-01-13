import re
from pypdf import PdfReader
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

def clean_text(text: str) -> str:
    """
    Remove sections like 'Bibliography' or 'References' if present.
    """
    match = re.search(r"(Bibliography|References)", text, re.IGNORECASE)
    return text[:match.start()] if match else text


def chunk_text(text: str, max_chunk_length: int = 2500) -> list:
    """
    Split text into smaller chunks; for RAG, shorter chunks are easier to retrieve.
    """
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 > max_chunk_length:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
        else:
            current_chunk += para + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def embed_chunks(chunks: list, embedder) -> np.ndarray:
    """
    Compute embedding for each chunk.
    """
    return np.array([embedder.encode(chunk) for chunk in chunks])


def retrieve_relevant_chunks(query: str, chunks: list, chunk_embeddings: np.ndarray,
                              embedder, top_k: int = 3) -> list:
    """
    Retrieve top_k chunks that are most similar to the query.
    """
    query_embedding = embedder.encode(query)
    norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    similarities = np.dot(chunk_embeddings, query_embedding) / (norms + 1e-10)
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


def rag_summarize(document_text: str, query: str) -> str:
    """
    Given a document and a query, retrieve top relevant chunks and use them to prompt the LLM.
    """
    cleaned_text = clean_text(document_text)
    chunks = chunk_text(cleaned_text)
    print(f"PDF split into {len(chunks)} chunks.")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_chunks(chunks, embedder)
    relevant_chunks = retrieve_relevant_chunks(query, chunks, embeddings, embedder, top_k=3)
    context = "\n".join(relevant_chunks)




    '''
    internal prompt:
    Question: <<Your question>>
    Context: <<input data e.g. pdf file content>>
    Additional instruction : Answer concisely based on the context
    Additional instruction : Answer in max 15 bullet points
    '''



    internal_prompt = (f"Question: {query}\n\nContext:\n{context}\n\n" +
                       "Answer concisely based on the context " +
                       "Answer in max 15 bullet points "
                       )
    response = ollama.generate(model="gpt-oss:120b-cloud", prompt=internal_prompt)
    return response.get("response", "").strip()


def begin_rag(text, query):
    """
    Process a file using RAG: read the file, summarize it
    """
    try:
        # get the answer for the intended question
        answer = rag_summarize(text, query)
        print(query)
        print("==========================================")
        print(answer)
        print("==========================================")
    except Exception as e:
        print(f"Error occurred")
        return None


def main():
    data_in_path = "C:\\Users\\malas\\shri_1000_names\\PythonProjects\\p13_rag1\\data_in\\ec2-types.pdf"
    text = ""
    try:
        # read all pdf content into a string
        reader = PdfReader(data_in_path)
        print(f"PDF page count {len(reader.pages)}")
        for nn in range(len(reader.pages)):
            page = reader.pages[nn]
            text = text + page.extract_text()
    except Exception as e:
        print(f"Error occurred")
        return None

    prompt_question_list = [
        "What does the instance type determine when launching an EC2 instance?",
        "How are resources like CPU and memory allocated on a host computer?",
        "What is the primary use case for General Purpose instance types?",
        "What is the difference between fixed and burstable performance instances?",
        "In the naming convention 'c7gn.4xlarge', what does the '7' represent?",
        "Which instance family is recommended for video encoding or high-volume websites?",
        "What do Flex instances like M7i-flex provide to a user?",
        "What does the 'd' suffix in an instance name (e.g., M5d) signify?",
        "How does EC2 handle shared resources when they are underused by other instances?",
        "Which high-memory instances are currently marked as no longer available for new launches?",
        "What is the recommended upgrade path for a user needing high-memory U-series instances?",
        "What specific hardware option does the 'a' represent in naming conventions?",
        "How does allocating a larger share of shared resources impact I/O performance?",
        "Which instance families are categorized under Accelerated Computing?",
        "What is the architectural difference indicated by 'g' versus 'i' in the instance series?"]

    print(f"\nBegin RAG.")
    for each_question in prompt_question_list:
        begin_rag(text, each_question)

    return 1

if __name__ == "__main__":
    main()