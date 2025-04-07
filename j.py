import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
import os
import tempfile
from datetime import datetime
import hashlib

# Load secrets from Streamlit
if "GROQ_API_KEY" not in st.secrets:
    st.error("Groq API key not found in secrets. Please configure your secrets.")
    st.stop()

if "HF_TOKEN" not in st.secrets:
    st.warning("Hugging Face token not found in secrets. Some models may not work without it.")
    hf_token = None
else:
    hf_token = st.secrets["HF_TOKEN"]
    os.environ['HF_TOKEN'] = hf_token

# Constants
DEFAULT_MODEL = "Llama3-70b-8192"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 500
TEMPERATURE = 0.2
MAX_TOKENS = 2048

# Initialize Embeddings with error handling
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
except Exception as e:
    st.error(f"Failed to load default embedding model: {str(e)}")
    st.warning("Falling back to alternative embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Alternative model name format
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# Streamlit UI Configuration
st.set_page_config(
    page_title="Advanced PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration
with st.sidebar:
    st.title("⚙ Settings & Configuration")
    
    # Model Selection
    model_options = {
        "Llama3-70b-8192": "Best quality (recommended)",
        "Llama3-8b-8192": "Faster with good quality",
        "Mixtral-8x7b-32768": "Large context window",
        "Gemma2-9b-It": "Fastest but lower quality"
    }
    selected_model = st.selectbox(
        "Select Groq Model",
        options=list(model_options.keys()),
        index=0,
        help="Select model based on your quality/speed needs"
    )
    
    # Advanced Options
    with st.expander("Advanced Options"):
        chunk_size = st.slider("Chunk Size", 1000, 10000, CHUNK_SIZE, 
                             help="Larger chunks capture more context but may reduce precision")
        chunk_overlap = st.slider("Chunk Overlap", 100, 2000, CHUNK_OVERLAP,
                                help="Helps maintain context between chunks")
        temperature = st.slider("Temperature", 0.0, 1.0, TEMPERATURE,
                              help="Lower for precise answers, higher for creativity")
        max_tokens = st.slider("Max Response Tokens", 100, 4096, MAX_TOKENS,
                              help="Maximum length of responses")
        search_k = st.slider("Document Chunks to Retrieve", 1, 10, 4,
                            help="Number of relevant document chunks to use for answers")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Advanced PDF Chat Assistant** features:
    - Multi-document analysis with deep context understanding
    - Conversation history across sessions
    - Precise answers with document references
    - Advanced document processing pipeline
    - Optimized for technical and complex documents
    """)

# Main UI
st.title("📚 Advanced PDF Chat Assistant")
st.markdown("""
Upload and interact with your PDF documents using state-of-the-art AI. 
Get accurate, context-aware answers with source citations.
""")

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'file_hashes' not in st.session_state:
    st.session_state.file_hashes = set()

# Generate session ID based on current time
def generate_session_id():
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Session Management
col1, col2 = st.columns(2)
with col1:
    session_id = st.text_input("Session ID", value=generate_session_id(),
                             help="Unique ID for your conversation session")
with col2:
    if st.button("🔄 New Session", help="Start a fresh conversation"):
        session_id = generate_session_id()
        st.session_state.store[session_id] = ChatMessageHistory()
        st.success(f"New session created: {session_id}")
        st.rerun()

# File processing functions
def get_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def process_pdf(file_path):
    try:
        # Try PyMuPDF first (faster and more reliable)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        st.warning(f"PyMuPDF failed, falling back to PyPDFLoader: {str(e)}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    return docs

# File Uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", 
                                 accept_multiple_files=True,
                                 help="Upload multiple PDFs for analysis")

if uploaded_files:
    with st.spinner("🔍 Processing and indexing documents..."):
        documents = []
        new_files = []
        
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.getvalue()
            file_hash = get_file_hash(file_content)
            
            if file_hash not in st.session_state.file_hashes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(file_content)
                    temp_path = temp_file.name
                
                try:
                    docs = process_pdf(temp_path)
                    documents.extend(docs)
                    st.session_state.file_hashes.add(file_hash)
                    new_files.append(uploaded_file.name)
                    os.unlink(temp_path)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue
        
        if documents:
            # Enhanced text splitting with metadata preservation
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
                add_start_index=True
            )
            
            splits = text_splitter.split_documents(documents)
            
            # Create or update FAISS vector store
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = FAISS.from_documents(splits, embedding=embeddings)
            else:
                st.session_state.vectorstore.add_documents(splits)
            
            st.success(f"✅ Processed {len(documents)} pages from {len(new_files)} new files")
            st.session_state.processed_files.update(new_files)

# Only proceed if API key is available and documents are processed
if groq_api_key and st.session_state.vectorstore:
    try:
        # Initialize Groq with enhanced settings
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=selected_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        # Configure retriever with MMR for diverse results
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": search_k,
                "fetch_k": min(20, search_k*3),
                "lambda_mult": 0.5
            }
        )
        
        # Enhanced Contextual Question Reformulation Prompt
        contextualize_q_system_prompt = """You are an expert at understanding and refining questions based on conversation history. 
        
        Your responsibilities:
        1. Analyze the full conversation history and current question
        2. Identify any implicit context, references, or pronouns
        3. Reformulate the question to be completely standalone while:
           - Preserving all original intent and nuance
           - Expanding any ambiguous references
           - Maintaining technical specificity
        4. Never answer the question - only clarify and expand it
        5. For follow-up questions, ensure connection to previous context is explicit
        """
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # Optimized QA System Prompt
        qa_system_prompt = """You are an expert research assistant with deep knowledge of the provided documents. Follow these guidelines:

1. **Answer Quality**:
   - Be precise, concise, and professional
   - Use academic tone for technical content
   - Break complex answers into logical paragraphs
   - Use bullet points for lists or comparisons

2. **Source Handling**:
   - ONLY use information from the provided context
   - Cite sources with exact page numbers when possible
   - If unsure, say "The documents don't contain this information"
   - For partial information, indicate what's available

3. **Context Awareness**:
   - Maintain conversation context
   - Recognize follow-up questions
   - Connect related concepts across documents

4. **Response Structure**:
   - Start with direct answer
   - Follow with supporting evidence
   - End with potential implications or connections

Context:
{context}

Current conversation:
{chat_history}

Question: {input}
        """
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Create chains with enhanced configuration
        question_answer_chain = create_stuff_documents_chain(
            llm, 
            qa_prompt,
            document_prompt=ChatPromptTemplate.from_template(
                "Document excerpt (Page {page}):\n{page_content}\n---"
            )
        )
        
        rag_chain = create_retrieval_chain(
            history_aware_retriever, 
            question_answer_chain
        )

        # Session history management
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Chat Interface
        st.markdown("---")
        st.subheader("💬 Document Analysis Chat")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing documents..."):
                    try:
                        session_history = get_session_history(session_id)
                        
                        response = conversational_rag_chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": session_id}}
                        )
                        
                        # Display answer
                        answer = response['answer']
                        st.markdown(answer)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Enhanced source display
                        if 'context' in response:
                            with st.expander("🔍 Detailed Source References"):
                                for i, doc in enumerate(response['context']):
                                    source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                                    page = doc.metadata.get('page', 'N/A')
                                    st.subheader(f"Source {i+1}: {source} (Page {page})")
                                    
                                    # Fixed score display
                                    score = doc.metadata.get('score')
                                    if isinstance(score, (float, int)):
                                        st.caption(f"Relevance score: {score:.2f}")
                                    else:
                                        st.caption("Relevance score: N/A")
                                    
                                    st.write(doc.page_content)
                                    st.markdown("---")
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

        # Chat history management
        st.markdown("---")
        with st.expander("📜 Session History Management"):
            if st.button("Clear Current Session History"):
                if session_id in st.session_state.store:
                    st.session_state.store[session_id].clear()
                    st.session_state.messages = []
                    st.success("Current session history cleared!")
                    st.rerun()
            
            if st.session_state.store.get(session_id):
                st.write(f"### Messages in session: {session_id}")
                for msg in st.session_state.store[session_id].messages:
                    st.text(f"{msg.type.upper()}: {msg.content}")
                
                # Export chat history
                export_text = "\n\n".join(
                    [f"{msg.type.upper()}:\n{msg.content}" 
                     for msg in st.session_state.store[session_id].messages]
                )
                
                st.download_button(
                    label="📥 Export Full Chat History",
                    data=export_text,
                    file_name=f"chat_history_{session_id}.txt",
                    mime="text/plain"
                )

    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
elif not groq_api_key:
    st.error("Please configure your GROQ_API_KEY in Streamlit secrets")
elif not st.session_state.vectorstore:
    st.info("Upload PDF documents to begin analysis. The system will process and index them for searching.")

# Document Management
with st.expander("📂 Document Management"):
    if st.session_state.processed_files:
        st.write("### Processed Documents:")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")
        
        if st.button("Clear All Documents"):
            st.session_state.vectorstore = None
            st.session_state.processed_files = set()
            st.session_state.file_hashes = set()
            st.success("All documents cleared. You can upload new files.")
            st.rerun()
    else:
        st.info("No documents currently processed")

# Footer
st.markdown("---")
st.caption("""
Advanced PDF Chat Assistant | Powered by Groq & LangChain | 
[Report Issues](https://github.com/your-repo/issues) | [Learn More](https://groq.com/)
""")