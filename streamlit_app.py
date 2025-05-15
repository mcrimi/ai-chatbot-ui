import streamlit as st
from openai import OpenAI
import os
from langchain.chat_models import ChatOpenAI
from langchain import hub
from langchain.docstore.document import Document
from langsmith import traceable, Client
from typing_extensions import List, TypedDict, Optional, Dict, Any
from langgraph.graph import START, StateGraph
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
import time
import pinecone
import json

#Keys
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "pathways-ai-assistant"

#Constants
DEFAULT_PINECONE_INDEX = "senegal-metrics-v1-1"
EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_NUM_DOCS = 4  # Default number of documents to retrieve


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.run_id_to_text = {}
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class State(TypedDict):
    question: str
    context: List[Document]
    doc_metadata: List[Dict[str, Any]]  # For storing document metadata
    answer: Optional[str]


def get_available_indexes():
    """Retrieve all available Pinecone indexes."""
    try:
        # Initialize Pinecone with newer SDK syntax
        pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        
        # List all indexes
        indexes = [index.name for index in pc.list_indexes()]
        
        # Make sure we have at least the default index in the list
        if not indexes:
            indexes = [DEFAULT_PINECONE_INDEX]
        elif DEFAULT_PINECONE_INDEX not in indexes:
            indexes.append(DEFAULT_PINECONE_INDEX)
            
        return indexes
    except Exception as e:
        st.error(f"Error retrieving Pinecone indexes: {e}")
        return [DEFAULT_PINECONE_INDEX]


def get_available_prompts():
    """Retrieve available prompts exclusively from Langsmith."""
    try:
        # Initialize Langsmith client
        client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
        
        # Get list of private prompts from Langsmith
        prompts = []
        
        # List all private prompts
        langsmith_prompts = client.list_prompts(is_public=False)
        
        # Extract prompt names and IDs
        if hasattr(langsmith_prompts, 'repos'):
            # Direct access to repos attribute if available
            for prompt in langsmith_prompts.repos:
                if hasattr(prompt, 'repo_handle') and hasattr(prompt, 'id'):
                    prompts.append({
                        'name': prompt.repo_handle,
                        'id': prompt.id
                    })
        else:
            # Fallback to handle different possible structures
            for prompt in langsmith_prompts:
                if hasattr(prompt, 'repo_handle') and hasattr(prompt, 'id'):
                    prompts.append({
                        'name': prompt.repo_handle,
                        'id': prompt.id
                    })
        
        if not prompts:
            st.warning("No prompts found in Langsmith. Please create some prompts in Langsmith first.")
            
        return prompts
            
    except Exception as e:
        st.error(f"Error retrieving prompts from Langsmith: {e}")
        return []


def get_prompt_content(prompt_name_or_id):
    """Get the content of a specific prompt from LangChain Hub using hub.pull.
    
    Args:
        prompt_name_or_id: Either a name (string) or a dictionary with 'name' and 'id' keys
    """
    try:
        # Determine the repo handle to use
        if isinstance(prompt_name_or_id, dict):
            repo_handle = prompt_name_or_id.get('name')
        else:
            # Legacy case: prompt_name_or_id is just the name as a string
            repo_handle = prompt_name_or_id
        
        if not repo_handle:
            st.warning("No prompt repo handle provided.")
            return ""
            
        # Use hub.pull to get the prompt content directly
        try:
            prompt_obj = hub.pull(repo_handle)
            if prompt_obj:
                # Convert to string for display
                return str(prompt_obj)
            else:
                st.warning(f"Prompt '{repo_handle}' could not be retrieved.")
                return ""
        except Exception as e:
            st.warning(f"Could not pull prompt '{repo_handle}': {e}")
            return ""
    
    except Exception as e:
        st.error(f"Error retrieving prompt from LangChain Hub: {e}")
        return ""


@traceable
def retrieve(state: State):
    # Get the currently selected index from session state
    current_index = st.session_state.get("selected_index", DEFAULT_PINECONE_INDEX)
    
    # Get number of documents to retrieve from session state
    num_docs = st.session_state.get("num_docs", DEFAULT_NUM_DOCS)
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    try:
        # Initialize Pinecone with newer SDK syntax
        pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(current_index)
        
        # Create vector store using the index
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        
        # Perform similarity search with the query
        retrieved_docs = vectorstore.similarity_search(state["question"], k=num_docs)
        
        # Log the number of documents retrieved
        st.session_state.last_retrieval_count = len(retrieved_docs)
        
        # Extract metadata from documents
        doc_metadata = []
        for doc in retrieved_docs:
            metadata = getattr(doc, 'metadata', {}) or {}
            # Add page content preview (first 100 chars)
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            metadata['content_preview'] = content_preview
            doc_metadata.append(metadata)
        
        # Store document metadata in session state for later display
        st.session_state.last_docs_metadata = doc_metadata
        
        return {
            "context": retrieved_docs,
            "doc_metadata": doc_metadata
        }
    except Exception as e:
        st.error(f"Error retrieving documents from index '{current_index}': {e}")
        empty_metadata = []
        return {
            "context": [Document(page_content=f"Error retrieving documents: {e}")],
            "doc_metadata": empty_metadata
        }


@traceable
def generate(state: State):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Get current selected prompt from session state
    current_prompt = None
    if st.session_state.selected_prompt:
        try:
            # Extract repo handle from the selected prompt
            if isinstance(st.session_state.selected_prompt, dict):
                repo_handle = st.session_state.selected_prompt.get('name')
            else:
                repo_handle = st.session_state.selected_prompt
                
            # Use hub.pull to get the prompt directly
            current_prompt = hub.pull(repo_handle)
            if not current_prompt:
                st.error(f"Selected prompt '{repo_handle}' could not be retrieved.")
                return {"answer": "Error: The selected prompt could not be retrieved. Please select a different prompt."}
        except Exception as e:
            st.error(f"Error retrieving prompt from LangChain Hub: {e}")
            return {"answer": f"Error retrieving prompt: {e}. Please select a different prompt or try again."}
    
    if not current_prompt:
        st.error("No valid prompt selected. Please select a valid prompt from LangChain Hub.")
        return {"answer": "Error: No valid prompt is selected. Please select a prompt from the sidebar."}
    
    # Format context from retrieved documents
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Get chat history from memory and convert to serializable format
    chat_history = st.session_state.memory.chat_memory.messages
    serialized_history = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            serialized_history.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serialized_history.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            serialized_history.append({"role": "system", "content": msg.content})
    
    # Simplified one-shot approach
    # Create messages based on prompt type
    if hasattr(current_prompt, "invoke"):
        try:
            # Check required variables for this template
            required_variables = getattr(current_prompt, 'input_variables', [])
            
            # Prepare variables based on template requirements
            if 'json_data' in required_variables:
                # Some templates expect JSON input
                json_data = {
                    "context": docs_content,
                    "question": state["question"],
                    "chat_history": serialized_history
                }
                input_variables = {"json_data": json.dumps(json_data)}
            elif 'context' in required_variables and 'question' in required_variables:
                # Standard variables
                input_variables = {
                    "context": docs_content,
                    "question": state["question"],
                    "chat_history": serialized_history
                }
            else:
                # Fall back to whatever variables the template expects
                input_variables = {}
                if 'context' in required_variables:
                    input_variables['context'] = docs_content
                if 'question' in required_variables:
                    input_variables['question'] = state["question"]
                if 'query' in required_variables:
                    input_variables['query'] = state["question"]
                if 'input' in required_variables:
                    input_variables['input'] = state["question"]
                if 'chat_history' in required_variables:
                    input_variables['chat_history'] = serialized_history
            
            # Invoke the template with our variables
            messages = current_prompt.invoke(input_variables)
        except Exception as e:
            # Fall back to basic format if template invocation fails
            st.warning(f"Error using template with invoke: {e}. Falling back to basic format.")
            
            # Simple system prompt + context + question format
            messages = [
                SystemMessage(content=str(current_prompt)),
                SystemMessage(content=f"Context information:\n{docs_content}"),
                *chat_history,  # Add chat history
                HumanMessage(content=state["question"])
            ]
    else:
        # For basic string prompts, use the traditional approach
        messages = [
            SystemMessage(content=str(current_prompt)),
            SystemMessage(content=f"Context information:\n{docs_content}"),
            *chat_history,  # Add chat history
            HumanMessage(content=state["question"])
        ]
    
    # Set up streaming
    stream_handler = StreamHandler(st.empty())
    
    # Create LLM with streaming
    llm = ChatOpenAI(
        model_name="gpt-4", 
        temperature=0.5, 
        openai_api_key=OPENAI_API_KEY,
        streaming=True,
        callbacks=[stream_handler]
    )
    
    # Generate response
    response = llm.invoke(messages)
    
    # Save the conversation to memory
    st.session_state.memory.save_context(
        {"input": state["question"]},
        {"output": response.content}
    )
    
    # Return the answer (the UI code will handle storing the message with metadata)
    return {"answer": response.content}
 

# Initialize session state for chat history (simplified, no memory)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize conversation memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Initialize session state for selected index
if "selected_index" not in st.session_state:
    st.session_state.selected_index = DEFAULT_PINECONE_INDEX

# Initialize session state for selected prompt
if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None  # Will be set to a dict with 'name' and 'id' keys when selected

# Initialize session state for number of documents to retrieve
if "num_docs" not in st.session_state:
    st.session_state.num_docs = DEFAULT_NUM_DOCS

# Initialize session state for tracking last retrieval count
if "last_retrieval_count" not in st.session_state:
    st.session_state.last_retrieval_count = None

# Initialize session state for tracking document metadata
if "last_docs_metadata" not in st.session_state:
    st.session_state.last_docs_metadata = []

# Initialize session state for tracking enhanced query
if "last_query" not in st.session_state:
    st.session_state.last_query = None

st.set_page_config(
    page_title="Segment Explorer AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': 'https://github.com/streamlit/streamlit/issues',
        'About': "### Segment Explorer AI Assistant"
    }
)

# Sidebar with configuration options
st.sidebar.title("Configuration")

# Pinecone index selection
st.sidebar.subheader("Knowledge Base")
available_indexes = get_available_indexes()
selected_index = st.sidebar.selectbox(
    "Select Pinecone Index",
    options=available_indexes,
    index=available_indexes.index(st.session_state.selected_index) if st.session_state.selected_index in available_indexes else 0
)

# Number of documents to retrieve
num_docs = st.sidebar.slider(
    "Number of documents to retrieve",
    min_value=2,
    max_value=20,
    value=st.session_state.num_docs,
    step=1,
    help="Adjust the number of documents retrieved from the vector database during search"
)

# Update session state if num_docs changed
if num_docs != st.session_state.num_docs:
    st.session_state.num_docs = num_docs
    st.sidebar.success(f"Will now retrieve {num_docs} documents per query.")

# Update session state if index changed
if selected_index != st.session_state.selected_index:
    st.session_state.selected_index = selected_index
    # Clear chat history when changing index
    st.session_state.messages = []
    st.sidebar.success(f"Switched to index: {selected_index}. Chat history cleared.")

# Prompt selection
st.sidebar.subheader("Assistant Behavior")
with st.sidebar.expander("Prompt Selection", expanded=True):
    try:
        available_prompts = get_available_prompts()
        
        if available_prompts:
            # Sort alphabetically for better usability
            available_prompts = sorted(available_prompts, key=lambda x: x['name'])
            
            # Extract names for display in selectbox
            prompt_names = [p['name'] for p in available_prompts]
            
            # If we have a stored prompt that's no longer available, reset it
            if st.session_state.selected_prompt is not None:
                selected_prompt_id = st.session_state.selected_prompt.get('id') if isinstance(st.session_state.selected_prompt, dict) else None
                if not any(p['id'] == selected_prompt_id for p in available_prompts):
                    st.session_state.selected_prompt = None
            
            # Default index is either the previously selected prompt or the first one
            default_index = 0
            if st.session_state.selected_prompt is not None and isinstance(st.session_state.selected_prompt, dict):
                try:
                    selected_name = st.session_state.selected_prompt.get('name')
                    if selected_name in prompt_names:
                        default_index = prompt_names.index(selected_name)
                except (ValueError, KeyError):
                    default_index = 0
            
            selected_name = st.selectbox(
                "Select System Prompt from Langsmith",
                options=prompt_names,
                index=default_index,
                help="Choose a prompt from your Langsmith prompts to define how the AI assistant behaves"
            )
            
            # Find the full prompt dictionary by name
            selected_prompt = next((p for p in available_prompts if p['name'] == selected_name), available_prompts[0])
            
            # Show info about the current prompt
            st.info(f"Using the '{selected_prompt['name']}' prompt from Langsmith.")
                
            # Update session state if prompt changed
            if st.session_state.selected_prompt != selected_prompt:
                st.session_state.selected_prompt = selected_prompt
                # Clear chat history when changing prompt
                st.session_state.messages = []
                st.success(f"Switched to prompt: {selected_prompt['name']}. Chat history cleared.")
                
            # Add button to view current prompt content
            if st.button("View Current Prompt"):
                try:
                    # Get the repo handle
                    repo_handle = selected_prompt['name']
                    
                    # Use hub.pull to get the prompt content directly
                    prompt_obj = hub.pull(repo_handle)
                    if prompt_obj:
                        # Convert to string format for display
                        prompt_content = str(prompt_obj)
                        st.code(prompt_content, language="markdown")
                    else:
                        st.warning(f"Prompt '{repo_handle}' has no content.")
                except Exception as e:
                    st.error(f"Error retrieving prompt content: {e}")
        else:
            st.error("No prompts available in Langsmith. Please create at least one prompt in Langsmith.")
                
    except Exception as e:
        st.error(f"Error loading prompts: {e}")

# Add a separate button to clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    st.sidebar.success("Chat history cleared.")

# Add advanced debugging options
with st.sidebar.expander("Advanced Options", expanded=False):
    # Option to show template information
    show_template_info = st.checkbox(
        "Show template information",
        value=False,
        help="Display information about the prompt template structure and variables"
    )
    
    st.session_state.show_template_info = show_template_info
    
    # Display template information if enabled and available
    if show_template_info and st.session_state.selected_prompt:
        st.divider()
        st.subheader("Template Information")
        
        try:
            # Extract repo handle
            if isinstance(st.session_state.selected_prompt, dict):
                repo_handle = st.session_state.selected_prompt.get('name')
            else:
                repo_handle = st.session_state.selected_prompt
                
            # Pull the prompt template
            prompt_obj = hub.pull(repo_handle)
            
            if prompt_obj:
                # Display template type
                template_type = type(prompt_obj).__name__
                st.write(f"**Type:** {template_type}")
                
                # Display required variables
                if hasattr(prompt_obj, 'input_variables'):
                    required_vars = prompt_obj.input_variables
                    st.write(f"**Required variables:** {', '.join(required_vars)}")
        except Exception as e:
            st.error(f"Error retrieving template info: {e}")

st.image("https://www.projectpathways.org/assets/images/common/logo/logo-color.svg", width=100)
# Show title and description.
st.title("✨Segment Explorer AI Assistant✨")
st.write(
    "Ask me questions about Senegal Segmentation Data and the Pathways Methodology."
)

# Display chat messages from history (keeping this for UI consistency)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If this is an assistant message with document metadata, display it
        if message["role"] == "assistant" and "doc_metadata" in message and message["doc_metadata"]:
            doc_metadata = message["doc_metadata"]
            with st.expander(f"📚 Source Documents ({len(doc_metadata)})"):
                for i, doc_meta in enumerate(doc_metadata):
                    doc_num = i + 1
                    st.markdown(f"**Document {doc_num}:**")
                    
                    # Display metadata in a clean format
                    if 'content_preview' in doc_meta:
                        st.markdown(f"*Preview:* {doc_meta['content_preview']}")
                        # Remove content_preview from display to avoid duplication
                        display_meta = {k: v for k, v in doc_meta.items() if k != 'content_preview'}
                    else:
                        display_meta = doc_meta
                    
                    # Display remaining metadata if any
                    if display_meta:
                        for key, value in display_meta.items():
                            st.markdown(f"*{key}:* {value}")
                    
                    # Add a separator between documents
                    if i < len(doc_metadata) - 1:
                        st.markdown("---")

# Show retrieval information if available
if st.session_state.last_retrieval_count is not None:
    retrieved_count = st.session_state.last_retrieval_count
    st.caption(f"Last query retrieved {retrieved_count} document{'s' if retrieved_count != 1 else ''} from the knowledge base.")

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Build the graph
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        
        # Invoke the graph with streaming - simplified without chat history
        result = graph.invoke({
            "question": prompt,
            "context": [],
            "doc_metadata": [],
            "answer": None
        })
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["answer"],
            "doc_metadata": st.session_state.last_docs_metadata.copy() if st.session_state.last_docs_metadata else []
        })
        
        # Display document metadata after response in an expander
        if st.session_state.last_docs_metadata:
            with st.expander(f"📚 Source Documents ({len(st.session_state.last_docs_metadata)})"):
                for i, doc_meta in enumerate(st.session_state.last_docs_metadata):
                    doc_num = i + 1
                    st.markdown(f"**Document {doc_num}:**")
                    
                    # Display metadata in a clean format
                    if 'content_preview' in doc_meta:
                        st.markdown(f"*Preview:* {doc_meta['content_preview']}")
                        # Remove content_preview from display to avoid duplication
                        display_meta = {k: v for k, v in doc_meta.items() if k != 'content_preview'}
                    else:
                        display_meta = doc_meta
                    
                    # Display remaining metadata if any
                    if display_meta:
                        for key, value in display_meta.items():
                            st.markdown(f"*{key}:* {value}")
                    
                    # Add a separator between documents
                    if i < len(st.session_state.last_docs_metadata) - 1:
                        st.markdown("---")
