import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import openai
import zipfile
import shutil
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Create an instance of the Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "codebase-rag"

# Check if index exists, create if it doesn't
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Replace with your embedding dimension
        metric='cosine',  # Or 'euclidean', depending on your use case
        spec=ServerlessSpec(
            cloud='gcp',  # Replace with your cloud provider
            region='us-west1'  # Replace with your region
        )
    )

# Connect to the index
pinecone_index = pc.Index(index_name)

# Define the embeddings
def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    if isinstance(text, list):
        return model.encode(text)
    else:
        return model.encode([text])[0]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Function to truncate text to a specific length to avoid token limits
def truncate_text(text, max_length=2000):
    """Truncate the input text if it exceeds the maximum length."""
    if len(text) > max_length:
        return text[:max_length] + "... (truncated)"
    return text

# Function to get file content
def get_file_content(file_path, repo_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Truncate content to avoid exceeding token limits
        content = truncate_text(content)

        # Get relative path from repo root
        rel_path = os.path.relpath(file_path, repo_path)

        return {
            "name": rel_path,
            "content": content
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

# Function to get main files content
SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                        '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}

def get_main_files_content(repo_path: str):
    files_content = []

    try:
        for root, dirs, files in os.walk(repo_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

            # Process each file in current directory
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)

    except Exception as e:
        print(f"Error reading repository: {str(e)}")

    return files_content

# Function to index documents into Pinecone
def index_documents(docs):
    vectors = []
    for doc in docs:
        content = doc['content']
        vector = get_huggingface_embeddings(content).tolist()
        metadata = {'text': content, 'source': doc['name']}
        id = doc['name']
        vectors.append((id, vector, metadata))

    # Upsert vectors to Pinecone
    batch_size = 100  # Adjust batch size as needed
    for i in range(0, len(vectors), batch_size):
        pinecone_index.upsert(vectors=vectors[i:i+batch_size])

# Function to perform RAG using Groq API
def perform_rag_without_streamlit(query):
    raw_query_embedding = get_huggingface_embeddings(query)

    top_k = 5
    results = pinecone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    # Get the list of retrieved texts
    contexts = []
    for match in results['matches']:
        source = match['metadata'].get('source', '')
        text = match['metadata'].get('text', '')
        contexts.append(f"Source: {source}\n{text}")

    augmented_context = "\n\n-------\n\n".join(contexts) + "\n\n-------\n\n"

    # Truncate the augmented context if it's too long
    augmented_context = truncate_text(augmented_context, max_length=3000)

    # Define the system prompt
    system_prompt = """
You are a Senior Software Engineer, specializing in TypeScript and JavaScript frameworks.
"""

    # Construct the user message
    user_message = f"""
I will provide you with code from a TypeScript file. Your task is to provide a detailed breakdown of the file, explaining how it works, the purpose of each function, the flow of data, and any advanced React or Convex techniques used.

Please break down the code into these sections:
1. File Overview: Summarize what the file does.
2. Component Analysis: Describe the main React component, how it interacts with the state, and what user interactions are handled.
3. Function Analysis: Describe any functions used, their purpose, and how they contribute to the overall file.
4. Hooks and State Management: Explain the hooks used in this file and their role in state management.
5. Potential Improvements: Provide suggestions for how this file could be improved.

Here is the file context:
<CONTEXT>
{augmented_context}
</CONTEXT>

MY QUESTION:
{query}
"""

    # Truncate the user message if it's too long
    user_message = truncate_text(user_message, max_length=4000)

    # Set Groq API key and base URL
    openai.api_key = GROQ_API_KEY
    openai.api_base = "https://api.groq.com/openai/v1"

    # Call Groq ChatCompletion API
    response = openai.ChatCompletion.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_message.strip()}
        ],
        max_tokens=1000,
        temperature=0.7
    )

    return response['choices'][0]['message']['content']
