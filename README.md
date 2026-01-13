# Ollama Unleashed: Interactive PDF RAG (Happy vs. Unhappy Path)

# YouTube video link : https://youtu.be/d-OcaC4SHJE

AI assisted Ollama based RAG for interactive PDF

Compare Cloud Ollama RAG response against Gemini in this deep-dive technical stress test. 
We analyse interactive PDF communication performance using happy and unhappy path queries to identify the superior architecture.

# Steps
As explained in above YouTube video, install Ollama (Windows/ Linux) -> Sign In

Download cloud AI model using: 
ollama run gpt-oss:120b-cloud

git clone https://github.com/imademethink/Ollama-based-RAG-Interactive-PDF.git

cd Ollama-based-RAG-Interactive-PDF-master

pip install -r requirements.txt

python p1_ollama_query.py

python p2_ollama_RAG_simple.py

python p3_ollama_RAG_Happy.py
