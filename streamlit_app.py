import streamlit as st
from openai import OpenAI
import os
from langchain.chat_models import ChatOpenAI
from langchain import hub
from langchain.docstore.document import Document
from langsmith import traceable, Client
from typing_extensions import List, TypedDict, Optional, Dict, Any, Sequence, Annotated
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain.callbacks.base import BaseCallbackHandler
import time
import pinecone
import json
import uuid
import atexit

# Global memory variable
memory = None

# Session management
def clear_session_memory():
    """Clear memory when session ends"""
    try:
        if 'thread_id' in st.session_state and memory is not None:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            memory.clear(config)
    except Exception as e:
        print(f"Error clearing session memory: {e}")

# Register cleanup function
atexit.register(clear_session_memory)

# Set page config first
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

#Keys
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "pathways-ai-assistant"

#Constants
DEFAULT_PINECONE_INDEX = "segment-explorer-v1-0"
EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_NUM_DOCS = 10  # Default number of documents to retrieve


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.run_id_to_text = {}
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Define our state schema for LangGraph
class ChatState(TypedDict):
    question: str
    context: List[Document]
    doc_metadata: List[Dict[str, Any]]
    messages: Annotated[Sequence[BaseMessage], add_messages]
    answer: Optional[str]
    last_retrieval_count: Optional[int]
    selected_index: str
    num_docs: int
    selected_prompt: Optional[Dict[str, str]]


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


def create_llm(stream_handler: StreamHandler) -> ChatOpenAI:
    """Create and return a configured ChatOpenAI instance."""
    return ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.5,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=True,
        callbacks=[stream_handler]
    )


@traceable
def retrieve(state: ChatState):
    # Get the latest question from messages
    latest_message = state["messages"][-1]
    if not isinstance(latest_message, HumanMessage):
        return state
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    try:
        # Initialize Pinecone with newer SDK syntax
        pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(state["selected_index"])
        
        # Create vector store using the index
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        
        # Perform similarity search with the query
        retrieved_docs = vectorstore.similarity_search(latest_message.content, k=state["num_docs"])
        
        # Extract metadata from documents
        doc_metadata = []
        for doc in retrieved_docs:
            metadata = getattr(doc, 'metadata', {}) or {}
            # Add page content preview (first 100 chars)
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            metadata['content_preview'] = content_preview
            doc_metadata.append(metadata)
        
        # Update state with retrieved information
        return {
            **state,
            "context": retrieved_docs,  # Store retrieved documents
            "doc_metadata": doc_metadata,
            "last_retrieval_count": len(retrieved_docs),
            "messages": state["messages"]  # Keep existing messages
        }
        
    except Exception as e:
        error_message = SystemMessage(content=f"Error retrieving documents: {e}")
        return {
            **state,
            "messages": state["messages"] + [error_message]
        }


@traceable
def generate(state: ChatState):
    # Get current selected prompt
    current_prompt = None
    if state["selected_prompt"]:
        try:
            # Extract repo handle from the selected prompt
            repo_handle = state["selected_prompt"].get('name')
            if not repo_handle:
                error_message = AIMessage(content="Error: The selected prompt could not be retrieved. Please select a different prompt.")
                return {**state, "messages": state["messages"] + [error_message]}
                
            # Use hub.pull to get the prompt directly
            current_prompt = hub.pull(repo_handle)
            if not current_prompt:
                error_message = AIMessage(content="Error: The selected prompt could not be retrieved. Please select a different prompt.")
                return {**state, "messages": state["messages"] + [error_message]}
        except Exception as e:
            error_message = AIMessage(content=f"Error retrieving prompt: {e}. Please select a different prompt or try again.")
            return {**state, "messages": state["messages"] + [error_message]}
    
    if not current_prompt:
        error_message = AIMessage(content="Error: No valid prompt is selected. Please select a prompt from the sidebar.")
        return {**state, "messages": state["messages"] + [error_message]}
    
    try:
        # Get the current question from the last human message
        current_question = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                current_question = msg
                break
        
        if not current_question:
            error_message = AIMessage(content="Error: No question found in the conversation.")
            return {**state, "messages": state["messages"] + [error_message]}
        
        # Create a minimal message history with:
        # 1. Last 3 message pairs (6 messages total)
        non_system_messages = [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]
        recent_messages = non_system_messages[-6:] if len(non_system_messages) > 6 else non_system_messages
        
        # Format context from retrieved documents
        context_content = "\n\n".join(doc.page_content for doc in state["context"]) if state["context"] else ""
        
        # Format the prompt with context and question
        formatted_prompt = current_prompt.format(
            context=context_content,
            question=current_question.content
        )
        
        # Create messages for the LLM
        messages = [SystemMessage(content=formatted_prompt)] + recent_messages
        
        # Create LLM for this turn
        stream_handler = StreamHandler(st.empty())
        llm = create_llm(stream_handler)
            
        # Generate response
        response = llm.invoke(messages)
        
        # Return updated state with the new response
        return {
            **state,
            "messages": state["messages"] + [response],
            "answer": response.content
        }
        
    except Exception as e:
        error_message = AIMessage(content=f"Error generating response: {e}")
        return {**state, "messages": state["messages"] + [error_message]}


# Create LangGraph workflow for chat
@st.cache_resource
def create_chat_graph():
    """Create and return the chat graph. This function is cached to ensure the graph is only created once."""
    global memory
    memory = InMemorySaver()
    workflow = StateGraph(ChatState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    return workflow.compile(checkpointer=memory)


# Create the graph once and cache it
graph = create_chat_graph()

# Streamlit UI code starts here

# Initialize session state for selected index
if "selected_index" not in st.session_state:
    st.session_state.selected_index = DEFAULT_PINECONE_INDEX

# Initialize session state for selected prompt
if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None  # Will be set to a dict with 'name' and 'id' keys when selected

# Initialize session state for number of documents to retrieve
if "num_docs" not in st.session_state:
    st.session_state.num_docs = DEFAULT_NUM_DOCS

# Create a unique thread ID for each user session if not exists
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Sidebar with configuration options
st.sidebar.title("Configuration")

# Prompt selection and Assistant Behavior section
st.sidebar.subheader("Assistant Behavior")
with st.sidebar.expander("Assistant Configuration", expanded=False):
    try:
        available_prompts = get_available_prompts()
        
        if available_prompts:
            # Sort alphabetically for better usability
            available_prompts = sorted(available_prompts, key=lambda x: x['name'])
            
            # Extract names for display in selectbox
            prompt_names = [p['name'] for p in available_prompts]
            
            # Set default index to 'response-generator' if it exists
            default_index = 0
            try:
                default_index = prompt_names.index('response-generator')
            except ValueError:
                # If 'response-generator' doesn't exist, use the first prompt
                default_index = 0
            
            # If we have a stored prompt that's no longer available, reset it
            if st.session_state.selected_prompt is not None:
                selected_prompt_id = st.session_state.selected_prompt.get('id') if isinstance(st.session_state.selected_prompt, dict) else None
                if not any(p['id'] == selected_prompt_id for p in available_prompts):
                    st.session_state.selected_prompt = None
                    # Reset to 'response-generator' if available
                    try:
                        default_index = prompt_names.index('response-generator')
                    except ValueError:
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

# Knowledge Base settings in a collapsed expander
with st.sidebar.expander("Knowledge Base Settings", expanded=False):
    # Pinecone index selection
    available_indexes = get_available_indexes()
    selected_index = st.selectbox(
        "Select Pinecone Index",
        options=available_indexes,
        index=available_indexes.index(st.session_state.selected_index) if st.session_state.selected_index in available_indexes else 0
    )

    # Number of documents to retrieve
    num_docs = st.slider(
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
        st.success(f"Will now retrieve {num_docs} documents per query.")

    # Update session state if index changed
    if selected_index != st.session_state.selected_index:
        st.session_state.selected_index = selected_index
        # Clear chat history when changing index
        st.session_state.messages = []
        st.success(f"Switched to index: {selected_index}. Chat history cleared.")

# Add a separate button to clear chat history
if st.sidebar.button("Clear Chat History"):
    # Clear the memory for the current thread
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        # Clear the memory for this thread
        if memory is not None:
            memory.clear(config)
            # Reset thread ID to create a new conversation thread
            st.session_state.thread_id = str(uuid.uuid4())
            st.sidebar.success("Chat history cleared.")
        else:
            st.sidebar.error("Memory not initialized. Please refresh the page.")
    except Exception as e:
        st.sidebar.error(f"Error clearing chat history: {e}")

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

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    # Initialize messages in session state if not exists
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display all messages in the chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Display assistant response
    with st.chat_message("assistant"):
        try:
            # Set up LangGraph config
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Get current state from memory
            current_state = graph.get_state(config)
            
            # Create initial state for this turn
            initial_state = {
                "question": prompt,
                "context": current_state.values.get("context", []) if current_state and hasattr(current_state, 'values') else [],
                "doc_metadata": current_state.values.get("doc_metadata", []) if current_state and hasattr(current_state, 'values') else [],
                "messages": (current_state.values.get("messages", []) if current_state and hasattr(current_state, 'values') else []) + [HumanMessage(content=prompt)],
                "answer": current_state.values.get("answer") if current_state and hasattr(current_state, 'values') else None,
                "last_retrieval_count": current_state.values.get("last_retrieval_count") if current_state and hasattr(current_state, 'values') else None,
                "selected_index": st.session_state.selected_index,
                "num_docs": st.session_state.num_docs,
                "selected_prompt": st.session_state.selected_prompt
            }
            
            # Invoke the graph
            result = graph.invoke(initial_state, config=config)
            
            # Add assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            
            # Display document metadata if available
            if result.get("doc_metadata"):
                doc_metadata = result["doc_metadata"]
                with st.expander(f"📚 Source Documents ({len(doc_metadata)})"):
                    for i, doc_meta in enumerate(doc_metadata):
                        doc_num = i + 1
                        st.markdown(f"**Document {doc_num}:**")
                        if 'content_preview' in doc_meta:
                            st.markdown(f"*Preview:* {doc_meta['content_preview']}")
                            display_meta = {k: v for k, v in doc_meta.items() if k != 'content_preview'}
                        else:
                            display_meta = doc_meta
                        if display_meta:
                            for key, value in display_meta.items():
                                st.markdown(f"*{key}:* {value}")
                        if i < len(doc_metadata) - 1:
                            st.markdown("---")
            
            # Show retrieval information if available
            if result.get("last_retrieval_count") is not None:
                retrieved_count = result["last_retrieval_count"]
                st.caption(f"Last query retrieved {retrieved_count} document{'s' if retrieved_count != 1 else ''} from the knowledge base.")
                
        except Exception as e:
            error_message = f"Error processing request: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
