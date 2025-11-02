# --- Standard library imports ---
import os  # For interacting with the operating system (e.g., setting environment variables, file paths)
import ssl  # For handling SSL/TLS security contexts
import tempfile

import boto3
# --- Third-party library imports ---
import certifi  # Provides Mozilla’s trusted CA bundle for SSL certificate verification
from dotenv import (
    load_dotenv,
)  # Loads environment variables from a .env file into the system environment

# --- LangChain core utilities ---
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # Splits text into smaller overlapping chunks for better embeddings
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    WebBaseLoader,
)

# --- LangChain document loaders ---
from langchain_core.documents import Document

# --- LangChain integrations (modern modular packages) ---
# from langchain_chroma import Chroma  # (Optional) For using Chroma as a local vector database
# from langchain_core.documents import Document  # Defines the Document structure (useful if constructing manually)
from langchain_openai import (
    OpenAIEmbeddings,
)  # Generates embeddings using OpenAI models (e.g., text-embedding-3-small)
from langchain_pinecone import (
    PineconeVectorStore,
)  # Integration for Pinecone vector database
from langchain_tavily import (
    TavilyCrawl,
)  # (Optional) Tavily crawl tool for web crawling

# A map of common file extensions to their corresponding LangChain loader
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    # ".docx": Docx2txtLoader, #not working as it needs extra dependency Docx2txt which is not added yet
    ".docx": UnstructuredWordDocumentLoader,
    # Add other loaders here (e.g., .csv, .json, etc.)
}

# --- Project-specific imports ---
from consts import INDEX_NAME  # Constant holding the name of the Pinecone index

def load_ssm_params(prefix="/askmydoc/"):
    """
    Load plain String parameters from AWS Systems Manager Parameter Store
    and set them as environment variables.
    """
    ssm = boto3.client("ssm", region_name="us-east-1")  # Update region if needed

    paginator = ssm.get_paginator("get_parameters_by_path")

    for page in paginator.paginate(Path=prefix, Recursive=True, WithDecryption=False):
        for param in page["Parameters"]:
            key = param["Name"].split("/")[-1]  # Extract last part, e.g. OPENAI_API_KEY
            value = param["Value"]
            os.environ[key] = value
            print(f"✅ Loaded {key} from AWS SSM")

    print("🎉 All SSM parameters loaded successfully.")

# --- Environment and SSL configuration ---
load_dotenv()  # Load environment variables (e.g., API keys) from .env file
load_ssm_params() # Load parameters from AWS SSM

# Configure SSL context to use certifi certificates for secure HTTPS requests
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# --- Embeddings and Vector Store Initialization ---
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)  # Initialize OpenAI embeddings model
# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)  # (Optional) Local Chroma DB
vectorstore = PineconeVectorStore(
    index_name="python-rag-index", embedding=embeddings
)  # Initialize Pinecone vector store
urls = ["https://python.langchain.com/en/latest/"]

all_docs = []
tavily_crawl = TavilyCrawl(api_key=os.getenv("TAVILY_API_KEY"))


def crawl_multiple(tavily_crawl, urls, **kwargs):
    all_docs = []
    for url in urls:
        # print("Crawling URL: ", url)
        result = tavily_crawl.invoke({"url": url, **kwargs})

        if isinstance(result, dict) and "results" in result:
            docs = result["results"]
        elif isinstance(result, list):
            docs = result
        else:
            docs = [result]

        # Convert to Document objects
        doc_objects = []
        for d in docs:
            if isinstance(d, str):
                doc_objects.append(Document(page_content=d))
            elif isinstance(d, dict) and "raw_content" in d:
                doc_objects.append(
                    Document(
                        page_content=d["raw_content"],
                        metadata={"source": d.get("url", url)},
                    )
                )

        print(f"  → Loaded {len(doc_objects)} docs from {url}")
        all_docs.extend(doc_objects)

    return all_docs


def ingest_docs(multi_urls: list):
    """
    Load, split, and ingest documentation into a Pinecone vector database.

    Workflow:
    1. Load documentation files from the local ReadTheDocs export.
    2. Split large documents into smaller overlapping text chunks.
    3. Normalize metadata (convert local file paths to valid URLs).
    4. Generate embeddings for each chunk using OpenAI’s embedding model.
    5. Insert the resulting vector data into Pinecone for semantic search and retrieval.
    """
    # --- Step 1: Load raw documentation ---
    print(f"........STEP 1 : LOADING DOCUMENTS STARTED......")
    # loader = ReadTheDocsLoader(path="langchain-docs/langchain.readthedocs.io/en/latest")
    # TavilyCrawl.invoke() expects a dict input
    docs = crawl_multiple(tavily_crawl, multi_urls, max_depth=1)
    print(f"✅ Loaded total {len(docs)} documents from Tavily")
    # print("********Final Documents: ", docs)
    if not docs:
        print("⚠️ No documents loaded — check TAVILY_API_KEY or URL accessibility.")
        return

    print(f"........STEP 2 : TEXT SPLITTING STARTED......")
    # --- Step 2: Split documents into manageable chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Each chunk will have ~1000 characters
        chunk_overlap=100,  # Overlap between chunks for context preservation
        separators=["\n\n", "\n", " ", ""],  # Hierarchy of text split points
    )
    chunks = text_splitter.split_documents(documents=docs)
    print(f"Split into {len(chunks)} text chunks")

    print(f"........STEP 3 : EMBEDDING AND STORING To VECTORSTORE STARTED.....")
    # --- Step 4: Embed and upload to Pinecone ---
    print(f"Preparing to insert {len(chunks)} chunks into Pinecone")
    # --- Step 3a: Re-initialize embeddings (optional, but clean) ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # --- Step 3b: Safety filter to avoid Pinecone 4MB limit ---
    MAX_BYTES = int(3.5 * 1024 * 1024)  # 3.5 MB limit per vector
    safe_documents = []
    for d in chunks:
        size = len(d.page_content.encode("utf-8"))
        if size > MAX_BYTES:
            print(
                f"⚠️ Skipping oversized chunk ({size / 1024:.1f} KB) from {d.metadata.get('source')}"
            )
        else:
            safe_documents.append(d)

    print(f"✅ {len(safe_documents)} safe chunks ready for upload to Pinecone")

    # --- Step 3c: Upload to Pinecone ---
    if not safe_documents:
        print("⚠️ No valid chunks to upload — check your chunk size or crawler content.")
        return

    # To avoid sending all docs in one large payload, upload in batches of 50 chunks
    BATCH_SIZE = 50
    print(f"Uploading {len(safe_documents)} documents in batches of {BATCH_SIZE}")

    for i in range(
        0, len(safe_documents), BATCH_SIZE
    ):  # i value increases by BATCH_SIZE like 0, 50, 100, ...
        batch = safe_documents[i : i + BATCH_SIZE]
        print(
            f"📤 Uploading batch {i // BATCH_SIZE + 1}/{(len(safe_documents) + BATCH_SIZE - 1) // BATCH_SIZE} "
            f"({len(batch)} docs)..."
        )
        PineconeVectorStore.from_documents(
            documents=batch, embedding=embeddings, index_name=INDEX_NAME
        )

    print("✅ Successfully added documents to Pinecone vector store")


# ----------------------------------------------------------------------
# 🕸️ Dynamic Webpage ingestion
# ----------------------------------------------------------------------
def ingest_webpage(url: str):
    """Crawl and ingest a webpage dynamically."""
    try:
        # --- Step 1: Load raw documentation ---
        print(f"........STEP 1 : LOADING DOCUMENTS STARTED......")
        # add code to split url if multiple urls are passed
        multi_urls = [url.strip() for url in url.split(",")]
        docs = crawl_multiple(tavily_crawl, multi_urls, max_depth=1)
        if not docs:
            return f"❌ No content found at {url}"

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        vectorstore.add_documents(chunks)

        return f"✅ Successfully ingested webpage: {url} ({len(chunks)} chunks)"
    except Exception as e:
        return f"⚠️ Error while ingesting webpage: {str(e)}"


# --- Entry point ---
if __name__ == "__main__":
    ingest_docs(urls)
