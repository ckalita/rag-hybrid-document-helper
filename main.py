import os
import re
from typing import Set

import streamlit as st

from ingestion_from_ui import ingest_file, ingest_webpage

# Assuming 'llm_call' and 'run_llm' are correctly set up
from llm_call import run_llm
from llm_hybrid_history import run_llm_hybrid

import html as _html  # for escaping text when rendering as HTML

# --- Custom CSS for wider sidebar and overall layout ---
st.markdown(
    """
    <style>
    /* Sidebar width */
    [data-testid="stSidebar"] {
        min-width: 400px;
        max-width: 420px;
    }

    /* Main content padding */
    .main {
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Improve sidebar scroll visibility */
    [data-testid="stSidebar"] section {
        overflow-y: auto;
        padding-right: 10px;
    }

    /* Optional: Slightly larger font for sidebar */
    [data-testid="stSidebar"] * {
        font-size: 15px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# --- Chat CSS (scrollable window + fixed input) ---
st.markdown(
    """
    <style>
    /* === CHAT CONTAINER STYLING === */
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        height: 85vh; /* Adjust height as needed */
        justify-content: space-between;
    }

    /* Scrollable area for messages */
    .chat-window {
        flex: 1;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f9fafb;
        margin-bottom: 10px;
    }

    /* Chat message bubbles */
    .user-message, .bot-message {
        max-width: 75%;
        padding: 10px 15px;
        border-radius: 15px;
        line-height: 1.5;
        font-size: 16px;
        word-wrap: break-word;
        margin: 4px 0;
        white-space: pre-wrap;
    }

    .user-message {
        align-self: flex-end;
        background-color: #DCF8C6;
        color: #000;
        border-top-right-radius: 0;
    }

    .bot-message {
        align-self: flex-start;
        background-color: #E5E5EA;
        color: #000;
        border-top-left-radius: 0;
    }

    /* Scrollbar styling */
    .chat-window::-webkit-scrollbar {
        width: 8px;
    }
    .chat-window::-webkit-scrollbar-thumb {
        background-color: #ccc;
        border-radius: 4px;
    }
    .chat-window::-webkit-scrollbar-thumb:hover {
        background-color: #aaa;
    }

    /* === FIXED INPUT BAR === */
    .fixed-input {
        position: sticky;
        bottom: 0;
        background: white;
        padding-top: 10px;
        padding-bottom: 6px;
        border-top: 1px solid #ddd;
    }

    /* Streamlit chat input tweaks - this targets streamlit input container */
    div[data-baseweb="input"] {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===================== FOOTER SECTION =====================
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 4px;  /* just beneath input field */
    left: 50%;
    transform: translateX(-50%);
    font-size: 12.5px;
    color: rgba(80, 80, 80, 0.65);  /* soft gray */
    background: rgba(255, 255, 255, 0.75);  /* translucent background */
    padding: 2px 10px;
    border-radius: 10px;
    backdrop-filter: blur(4px);  /* slight blur behind it */
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
    text-align: center;
    z-index: 9999;
    transition: opacity 0.3s ease;
}
.footer:hover {
    opacity: 0.95;  /* slightly brighten when hovered */
}
</style>
""", unsafe_allow_html=True)
st.markdown(
    """
    <div class="footer">
        © 2025 <b>Chandan Kalita</b> | Document Helper Chat Assistant 
        Built with ❤️ using <a href="https://streamlit.io" target="_blank">Streamlit</a> <a href="https://docs.langchain.com/oss/python/langchain/overview" target="_blank">Langchain</a> 
        <a href="https://www.pinecone.io/" target="_blank">Pinecon</a> & <a href="https://openai.com" target="_blank">OpenAI</a>
    </div>
    """,
    unsafe_allow_html=True
)


# ===================== FIXED CHATBOT HEADER =====================

# Inject CSS for the fixed header
st.markdown("""
<style>
/* --- FIXED HEADER --- */
.fixed-header {
    position: fixed;
    top: 48px;  /* push below the top system bar */
    left: calc(50% + 130px);  /* offset for sidebar */
    transform: translateX(-50%);
    z-index: 9999;
    width: clamp(420px, 62%, 950px);
    background: rgba(255, 255, 255, 0.98);
    border-radius: 8px;
    padding: 6px 10px;  /* keeps height small */
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center; /* vertically center content */
}

/* --- HEADER TEXT --- */
.header-title {
    font-weight: 700;
    font-size: 16px;
    color: #1E40AF;
    margin: 0;
    text-align: center;
    line-height: 1.1;
}

.header-desc {
    font-size: 15px;
    color: #333;
    line-height: 1.2;
    text-align: center;
    margin-top: 2px;
}

/* Adjust main padding to prevent overlap */
.stApp > main {
    padding-top: 85px !important; /* matches header height + margin */
}

/* --- RESPONSIVE --- */
@media (max-width: 800px) {
  .fixed-header {
    top: 60px;
    left: 50%;
    transform: translateX(-50%);
    width: calc(100% - 40px);
  }
}
</style>
""", unsafe_allow_html=True)





# Function to render the fixed header
def render_fixed_header(description_text: str):
    header_html = f"""
    <div class='fixed-header'>
        <div class='fixed-header-inner'>
            <div class='header-desc'>{description_text}</div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


# Initialize once and render header
if "app_description" not in st.session_state:
    st.session_state["app_description"] = (
        """
            <h3 style='text-align: center; color: #1E40AF;'>📚 Document Helper Chat Assistant</h3>
            Welcome! I’m here to help you with intelligent Q&A using your uploaded documents/webpages Links. <br>
            You can upload files, URLs, and chat naturally — all in one place.
            """
    )

render_fixed_header(st.session_state["app_description"])

# ================================================================
# Set a title for the app
st.set_page_config(page_title="Document Helper", page_icon="📘")
# --- Header and Description Section ---

st.sidebar.markdown(
    """
    **Enhance your assistant by adding your own content!**

    ✅ **Supported formats:** PDF, DOCX, TXT  
    🌐 **URLs:** Crawl any webpage for knowledge

    🤖 Once ingested, the assistant will use your uploaded content to give context-aware answers.

    💡 **Example questions:**
    - “Summarize the key points from my document”
    - “What does section 2 of the uploaded file explain?”
    """,
)

# --- Webpage Ingestion ---
web_urls = st.sidebar.text_input(
    label="",
    placeholder="Enter webpage URL(s), For multiple separated by commas...",
    key="web_urls",
)
if st.sidebar.button("Ingest Webpage"):
    if web_urls:
        with st.sidebar:
            with st.spinner("🔍 Crawling and embedding webpage..."):
                print("*******Web URL to ingest:", web_urls)  # Debugging line
                result = ingest_webpage(web_urls)
            st.sidebar.success(result)
            st.session_state["web_urls"] = ""
    else:
        st.sidebar.warning("Please enter a valid URL.")

# --- File Upload Ingestion ---
uploaded_file = st.sidebar.file_uploader(
    "Upload document (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"]
)
if uploaded_file:
    with st.sidebar:
        with st.spinner("📄 Processing file..."):
            print("*******Uploaded file to ingest:", uploaded_file.name)  # Debugging line
            result = ingest_file(uploaded_file)
        st.sidebar.success(result)

# --- Session State Initialization ---
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []  # LangChain history format

# --- Helper Functions ---
def create_sources_string(source_urls: Set[str]) -> str:
    """Formats a set of source URLs into a numbered markdown string."""
    if not source_urls:
        return ""

    sources_list = sorted(list(source_urls))  # Sort for consistent order
    sources_string = "\n\n**Sources:**\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i + 1}. {source}\n"
    return sources_string


def _to_html_safe(text: str) -> str:
    """
    Escape HTML special chars and convert newlines to <br> so messages render nicely
    when using unsafe_allow_html=True.
    """
    if text is None:
        return ""
    escaped = _html.escape(str(text))
    return escaped.replace("\n", "<br>")

# --- Render chat messages only if they exist ---
if "messages" in st.session_state and st.session_state["messages"]:
    st.markdown("<div class='chat-window'>", unsafe_allow_html=True)

    for message in st.session_state["messages"]:
        role_class = "user-message" if message["role"] == "user" else "bot-message"
        st.markdown(
            f"<div class='{role_class}'>{_html.escape(message['content'])}</div>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


# Auto-scroll to bottom on render
st.markdown(
    """
<script>
    (function() {
        var chatWindow = document.getElementById('chat-window');
        if (chatWindow) {
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
    })();
</script>
""",
    unsafe_allow_html=True,
)


# --- Fixed Input Bar ---
st.markdown("<div class='fixed-input'>", unsafe_allow_html=True)

# Use st.chat_input (keeps original UX)
prompt = st.chat_input("Enter your prompt here...")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # close chat-wrapper

# --- Handling User Input and calling LLM (keeps your original logic, only adapted to new rendering) ---
if prompt:
    # === Immediately show the user message ===
    st.session_state["user_prompt_history"].append(prompt)
    st.session_state["chat_history"].append(("human", prompt))

    # Display the updated chat window including the user's latest message right away
    chat_html = "<div class='chat-window' id='chat-window'>"

    for user_query, generated_response in zip(
        st.session_state["user_prompt_history"], st.session_state["chat_answers_history"]
    ):
        chat_html += f"<div class='user-message'>{_to_html_safe(user_query)}</div>"
        chat_html += f"<div class='bot-message'>{_to_html_safe(generated_response)}</div>"

    # Add the current (just-submitted) user message without a bot reply yet
    chat_html += f"<div class='user-message'>{_to_html_safe(prompt)}</div>"
    chat_html += "</div>"

    st.markdown(chat_html, unsafe_allow_html=True)

    # Auto-scroll right away
    st.markdown(
        """
        <script>
            var chatWindow = document.getElementById('chat-window');
            if (chatWindow) chatWindow.scrollTop = chatWindow.scrollHeight;
        </script>
        """,
        unsafe_allow_html=True,
    )

    # === Generate and show assistant response ===
    with st.spinner("🤖 Generating response..."):
        generated_response = run_llm_hybrid(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        print("*******Generated Response:", generated_response)  # Debugging line

        # ===================== FETCH THE SOURCES =====================
        try:
            sources = set([doc.metadata["source"] for doc in generated_response.get("context", [])])
        except Exception:
            sources = set()

        answer = generated_response.get("answer", "")
        is_simple_query = bool(
            re.match(
                r"^(hi|hello|hey|greetings|how are you|hii)\W*$",
                prompt.strip(),
                re.IGNORECASE,
            )
        )

        if "general_chat" in generated_response.get("answer_type", "") or is_simple_query:
            clean_answer = answer.replace("**NO_DOC_ANSWER**", "").strip()
            source_string = ""
        else:
            clean_answer = answer
            source_string = create_sources_string(sources) # format sources


        formatted_response = f"{clean_answer} {source_string}".strip()

        # Show assistant reply immediately
        st.markdown(
            f"<div class='bot-message'>{_to_html_safe(formatted_response)}</div>",
            unsafe_allow_html=True,
        )

        # Update histories
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("ai", generated_response.get("answer", "")))

        # Scroll again after the bot reply
        st.markdown(
            """
            <script>
                var chatWindow = document.getElementById('chat-window');
                if (chatWindow) chatWindow.scrollTop = chatWindow.scrollHeight;
            </script>
            """,
            unsafe_allow_html=True,
        )
    # After appending, Streamlit will rerun and the chat window will show the new messages

