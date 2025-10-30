from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI

# ----------------------------
# LLM Initialization
# ----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ----------------------------
# Prompt Template
# ----------------------------
RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant. Use the below context to answer the user question.
If the context doesn’t contain enough information, use your general knowledge to give a helpful and accurate answer.

Context:
{context}

Question:
{question}

Answer:
"""

rag_prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"]
)


# ----------------------------
# Generate Response Function
# ----------------------------
def generate_response(query, retriever):
    """
    Generate response using RAG (context + LLM), with fallback to normal chat if context is weak.
    """
    # --- Step 1: Retrieve context documents ---
    try:
        context_docs = retriever.invoke(query)
    except Exception as e:
        print(f"Retriever error: {e}")
        context_docs = []

    # Combine text from context documents
    context_text = "\n\n".join(
        [doc.page_content for doc in context_docs if isinstance(doc, Document)]
    )

    # --- Step 2: Decide if context is strong enough ---
    if not context_text.strip() or len(context_text) < 20:
        print("⚠️ Context too small, switching to general chat mode.")
        response = llm.invoke(query)
        return {
            "answer": (
                response.content if hasattr(response, "content") else str(response)
            ),
            "source_documents": [],
        }

    # --- Step 3: Ask with RAG template ---
    prompt = rag_prompt.format(context=context_text, question=query)
    response = llm.invoke(prompt)

    # --- Step 4: Check if LLM gave a non-informative answer ---
    if (
        "NO_DOC_ANSWER" in response.content.upper()
        or "I DON'T KNOW" in response.content.upper()
    ):
        print("ℹ️ RAG returned NO_DOC_ANSWER — retrying as general chatbot.")
        response = llm.invoke(query)
        return {
            "answer": (
                response.content if hasattr(response, "content") else str(response)
            ),
            "source_documents": [],
        }

    # --- Step 5: Normal successful RAG response ---
    return {
        "answer": response.content if hasattr(response, "content") else str(response),
        "source_documents": context_docs,
    }
