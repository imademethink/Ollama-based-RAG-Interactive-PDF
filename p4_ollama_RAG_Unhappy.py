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
        "How do I calculate the estimated monthly cost for an m5.large instance?",
        "What are the step-by-step instructions to launch an instance using the AWS Management Console?",
        "How do I configure a Security Group to allow SSH access to my instance?",
        "Which instance type is best for a specific version of a third-party software like SAP HANA?",
        "How do I create an Amazon Machine Image (AMI) from an existing instance?",
        ]

    print(f"\nBegin RAG.")
    for each_question in prompt_question_list:
        begin_rag(text, each_question)

    return 1

if __name__ == "__main__":
    main()