import os
from typing import Any, Dict, List

import boto3
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME

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

load_dotenv()
load_ssm_params()

def run_llm_hybrid(query: str, chat_history: List[Dict[str, Any]] = []):
    """
    Hybrid LLM pipeline:
    - Retrieves context from Pinecone
    - If relevant context found → use RAG
    - Otherwise → fall back to general chat
    """

    print("\n******************* inside run_llm_hybrid *******************\n")

    # --- Initialize embeddings and Pinecone vector store ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # --- Chat model configuration ---
    chat = ChatOpenAI(model="gpt-4o-mini", verbose=True, temperature=0.3)

    # --- Retrieve documents manually with score filtering ---
    k = 4
    score_threshold = 0.50

    print(f"🔍 Performing similarity search (k={k}, score_threshold={score_threshold})")
    docs_and_scores = docsearch.similarity_search_with_score(query, k=k)
    print(f"Retrieved {len(docs_and_scores)} docs (unfiltered).")
    print(f"***Retrieved docs_and_scores : ", docs_and_scores)

    # Adjust score threshold if any doc is from non-HTTP source
    score_threshold = (
        0.35
        if any(
            d.metadata.get("source", "") and "http" not in d.metadata.get("source", "")
            for d, _ in docs_and_scores
        )
        else score_threshold
    )

    # --- Filter documents by score threshold ---
    # retrieved_docs = [doc for doc, score in docs_and_scores if score >= score_threshold]
    filtered_docs_and_scores = [
        (doc, score) for doc, score in docs_and_scores if score >= score_threshold
    ]
    retrieved_docs = [doc for doc, _ in filtered_docs_and_scores]
    print(f"Retrieved {len(retrieved_docs)} docs after (filtered).")
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    context_strength = len(context_text.strip())

    print(f"Retrieved context length: {context_strength}")
    if not retrieved_docs:
        print("⚠️ No relevant documents retrieved — will use general chat mode.")
    else:
        print(f"✅ Retrieved {len(retrieved_docs)} relevant documents:")
        for i, (doc, score) in enumerate(filtered_docs_and_scores):
            print(
                f"   {i+1}. Score: {score:.3f} | Source: {doc.metadata.get('source', '')[:100]}"
            )

    # --- Check if context is meaningful ---
    has_relevant_context = len(retrieved_docs) > 0

    # --- Shared RAG prompt ---
    template_content = (
        "You are a precise assistant for answering questions using the provided context.\n\n"
        "If the answer can be found (even partially) in the context, answer strictly from that context.\n"
        "If the context does NOT contain enough relevant information, "
        "answer from your general knowledge and append exactly **NO_DOC_ANSWER** at the end.\n\n"
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

    # --- Case 1: No relevant context → General Chat ---
    if not has_relevant_context:
        print("💬 Switching to general chat mode (weak or no context).")

        general_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful and conversational assistant.\n\nContext: {context}",
                ),
                ("human", "{input}"),
            ]
        )

        general_chain = create_stuff_documents_chain(chat, general_prompt)
        general_result = general_chain.invoke(
            {
                "input": query,
                "context": "",
                "chat_history": chat_history,
            }
        )
        print("✅ General chat answer generated: ", general_result)
        result = {
            "answer": general_result,
            "context": [],
            "answer_type": "general_chat",
        }
        return result

    # --- Case 2: Relevant context → RAG Mode ---
    print("📚 Using RAG mode with filtered context (no redundant re-retrieval).")

    rag_input = {
        "input": query,
        "context": retrieved_docs,
        "chat_history": chat_history,
    }

    rag_result = stuff_documents_chain.invoke(rag_input)
    print("✅ RAG answer generated: ", rag_result)
    if isinstance(rag_result, dict) and "answer" in rag_result:
        answer = rag_result["answer"]
    else:
        answer = rag_result  # in case chain returns a string

    # --- Post-process: handle NO_DOC_ANSWER ---
    if "NO_DOC_ANSWER" in answer:
        print("⚠️ Model added NO_DOC_ANSWER — context may be weak.")
        answer = answer.replace("**NO_DOC_ANSWER**", "").strip()
    else:
        print("✅ Answer confidently derived from context.")

    result = {
        "answer": answer,
        "context": retrieved_docs,
        "answer_type": "RAG",
    }

    return result
