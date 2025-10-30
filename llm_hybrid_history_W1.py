from dotenv import load_dotenv
from typing import Any, Dict, List

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from consts import INDEX_NAME

load_dotenv()


def run_llm_hybrid(query: str, chat_history: List[Dict[str, Any]] = []):
    """
    Run a question-answering chain with LangChain + Pinecone + OpenAI.
    Falls back to normal chatbot mode when no relevant context is found.
    """
    print("*******************inside run_llm_hybrid*******************\n\n")
    # --- Initialize embeddings and vector store ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # --- LLM configuration ---
    chat = ChatOpenAI(model="gpt-4o-mini", verbose=True, temperature=0.2)

    # --- History-aware retriever for better query rewriting ---
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retriever = docsearch.as_retriever(search_kwargs={"k": 4})
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=retriever, prompt=rephrase_prompt
    )

    # --- Retrieve docs first for confidence logic ---
    retrieved_docs = retriever.invoke(query)
    context_text = " ".join([d.page_content for d in retrieved_docs])
    context_strength = len(context_text.strip())
    print("Retrieved context length:", context_strength)
    print("context_text:", context_text)
    # --- Template that enforces precise NO_DOC_ANSWER usage ---
    template_content = (
        "You are a precise assistant for answering questions using provided context.\n\n"
        "You will be given a question and a set of retrieved context passages.\n"
        "If the answer to the question can be found (even partially) in the provided context, "
        "you MUST answer STRICTLY based on that context and DO NOT add **NO_DOC_ANSWER**.\n\n"
        "ONLY if the context does NOT contain enough relevant information to answer, "
        "then answer from your general knowledge and append exactly **NO_DOC_ANSWER** at the END.\n\n"
        "Be absolutely certain before appending **NO_DOC_ANSWER**.\n\n"
        "Context:\n{context}\n"
    )

    retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template_content),
            ("human", "{input}"),
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa_chain = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    # --- Confidence-based fallback ---
    if context_strength < 500:  # Too little retrieved data
        print("⚠️ Weak or empty context detected — switching to general chat mode.")
        general_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant. Answer conversationally.\n\nContext: {context}"),
                ("human", "{input}"),
            ]
        )

        general_chain = create_stuff_documents_chain(chat, general_prompt)
        general_result = general_chain.invoke({"input": query, "context": ""})
        result = {
            "answer": general_result,
            "source_documents": [],
            "context": [],
            "answer_type": "general_chat"
        }
        return result

    print("Going to RAG mode with retrieved context.")
    # --- Standard retrieval-based QA ---
    result = qa_chain.invoke(input={"input": query, "chat_history": chat_history})

    # --- Post-process: remove NO_DOC_ANSWER if context was strong ---
    if "NO_DOC_ANSWER" in result.get("answer", "") and context_strength > 500:
        result["answer"] = result["answer"].replace("**NO_DOC_ANSWER**", "").strip()

    result["source_documents"] = retrieved_docs
    result["answer_type"] = "rag"
    return result
