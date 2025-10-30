import os
import re
from typing import Set

import streamlit as st

from ingestion_from_ui import ingest_file, ingest_webpage

# Assuming 'llm_call' and 'run_llm' are correctly set up
from llm_call import run_llm
from llm_hybrid_history import run_llm_hybrid


# --- Function to load external CSS ---
def local_css(file_name):
    """Reads a local CSS file and injects it into Streamlit."""
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # A simple st.error helps if the user accidentally deletes the file
        st.error(
            f"Error: Could not find CSS file: {file_name}. Check your folder structure."
        )


# 1. Load the CSS from the external file at the start
local_css("style.css")

# Set a title for the app
st.set_page_config(page_title="Document Helper", page_icon="📘")
st.title("📚 Document Helper Chatbot")

st.sidebar.header("🧠 Ingestion Options")

# --- Webpage Ingestion ---
web_urls = st.sidebar.text_input("Enter webpage URL to ingest:")
if st.sidebar.button("Ingest Webpage"):
    if web_urls:
        with st.sidebar:
            with st.spinner("🔍 Crawling and embedding webpage..."):
                print("*******Web URL to ingest:", web_urls)  # Debugging line
                result = ingest_webpage(web_urls)
            st.sidebar.success(result)
    else:
        st.sidebar.warning("Please enter a valid URL.")

# --- File Upload Ingestion ---
uploaded_file = st.sidebar.file_uploader(
    "Upload document (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"]
)
if uploaded_file:
    with st.sidebar:
        with st.spinner("📄 Processing file..."):
            print(
                "*******Uploaded file to ingest:", uploaded_file.name
            )  # Debugging line
            result = ingest_file(uploaded_file)
        st.sidebar.success(result)
# --- Session State Initialization ---

# Initialize chat history lists in session state if they don't exist
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = (
        []
    )  # LangChain's specific history format (e.g., [("human", "query"), ("ai", "answer")])


# --- Helper Function ---
def create_sources_string(source_urls: Set[str]) -> str:
    """Formats a set of source URLs into a numbered string."""
    if not source_urls:
        return ""
    sources_list = sorted(list(source_urls))  # Sort for consistent order

    # Use a cleaner format for the source string
    sources_string = "\n\n**Sources:**\n"
    for i, source in enumerate(sources_list):
        # Using markdown link format for easier clicking
        sources_string += f"{i + 1}. [{source}]({source})\n"
    return sources_string


# --- Displaying Chat History ---

# Iterate through the history and display messages
# It's cleaner to iterate over a single range/zip if all lists have the same length
for user_query, generated_response in zip(
    st.session_state["user_prompt_history"], st.session_state["chat_answers_history"]
):
    with st.chat_message("user"):
        st.write(user_query)
    with st.chat_message("assistant"):
        st.write(generated_response)

# --- Handling User Input ---

# Use st.chat_input which is designed for this purpose
# The prompt variable now holds the user's input only when they hit Enter or click send
if prompt := st.chat_input("Enter your prompt here..."):
    # Immediately display the user's new message
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            # Call your LLM function
            generated_response = run_llm_hybrid(
                query=prompt, chat_history=st.session_state["chat_history"]
            )

            print("*******Generated Response:", generated_response)  # Debugging line

            # Extract sources and create the final formatted response
            sources = set(
                [doc.metadata["source"] for doc in generated_response["context"]]
            )

            answer = generated_response["answer"]

            # 🌟 LOGIC: Determine if the answer is generic/greeting or a fallback
            # 1. Check for Simple Queries (Handles "Hii", "Hello", etc.)
            is_simple_query = bool(
                re.match(
                    r"^(hi|hello|hey|greetings|how are you|hii)\W*$",
                    prompt.strip(),
                    re.IGNORECASE,
                )
            )

            # 2. Check for the LLM's explicit fallback flag
            is_doc_fallback = "**NO_DOC_ANSWER**" in answer

            if (
                "general_chat" in generated_response.get("answer_type", "")
                or is_simple_query
            ):
                # If it's a greeting OR a known fallback, remove the flag and exclude sources
                clean_answer = answer.replace("**NO_DOC_ANSWER**", "").strip()
                source_string = ""
            else:
                # This is a genuine document-based answer, so we include sources
                clean_answer = answer
                source_string = create_sources_string(sources)

            formatted_response = f"{clean_answer} {source_string}"

            # Display the final response
            st.write(formatted_response)

        # Update session state history lists
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)

        # Update LangChain's history format
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["answer"]))
