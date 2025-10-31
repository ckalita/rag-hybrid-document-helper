# --- Standard library imports ---
import os  # For interacting with the operating system (e.g., setting environment variables, file paths)
import ssl  # For handling SSL/TLS security contexts
import tempfile

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
    WebBaseLoader,
    UnstructuredWordDocumentLoader,
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
from ingestion import ingest_docs

# --- Environment and SSL configuration ---
load_dotenv()  # Load environment variables (e.g., API keys) from .env file

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

all_docs = []
tavily_crawl = TavilyCrawl(api_key=os.getenv("TAVILY_API_KEY"))


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
        ingest_docs(multi_urls)
        return f"✅ Successfully ingested url(s): {url}"
    except Exception as e:
        return f"⚠️ Error while ingesting webpage: {str(e)}"


# ----------------------------------------------------------------------
# 📄 File ingestion (PDF, DOCX, TXT)
# ----------------------------------------------------------------------
def ingest_file(uploaded_file):
    """Handle file upload ingestion for PDF, DOCX, TXT."""
    print("*******INGEST FILE STARTED*******:", uploaded_file.name)
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    print("*******FILE SUFFIX*******:", suffix)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        # 1. Load Documents
        try:
            loader = get_document_loader(tmp_path)
            docs = loader.load()
        except ValueError as e:
            print(f"Error loading document: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during loading: {e}")
            return None

        # docs = loader.load()
        if not docs:
            return "⚠️ No readable content found in the file."

        # 2. Split Documents into Chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        print(f"Document split into {len(chunks)} chunks.")
        print("*******CHUNKS*******:", chunks)
        # 3. Index Chunks into Pinecone
        # Use the from_documents static method for a direct, one-step creation/addition
        PineconeVectorStore.from_documents(
            documents=chunks, embedding=embeddings, index_name=INDEX_NAME
        )

        return f"✅ Successfully ingested {uploaded_file.name}"
    except Exception as e:
        return f"⚠️ Error while processing {uploaded_file.name}: {str(e)}"


def get_document_loader(file_path: str):
    """Determines the correct LangChain DocumentLoader based on file extension."""
    ext = os.path.splitext(file_path)[-1].lower()
    loader_class = LOADER_MAPPING.get(ext)

    if loader_class:
        # Loaders are initialized with the file path
        return loader_class(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
