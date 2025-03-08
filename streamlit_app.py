import streamlit as st
from openai import OpenAI
import os
from langchain.chat_models import ChatOpenAI
from langchain import hub
from langchain.docstore.document import Document
from langsmith import traceable
from typing_extensions import List, TypedDict, Optional, Dict, Any
from langgraph.graph import START, StateGraph
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
import time
import pinecone

#Keys
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "pathways-ai-assistant"

#Constants
DEFAULT_PINECONE_INDEX = "senegal-metrics-v1-1"
EMBEDDING_MODEL = "text-embedding-3-large"
SYSTEM_PROMPT = """You are an AI assistant specialized in Senegal Segmentation Data and the Pathways Methodology. 
Answer questions based on the provided context. If you don't know the answer, say so instead of making up information."""


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
    chat_history: List[Dict[str, Any]]
    context: List[Document]
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


@traceable
def retrieve(state: State):
    # Get the currently selected index from session state
    current_index = st.session_state.get("selected_index", DEFAULT_PINECONE_INDEX)
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    try:
        # Initialize Pinecone with newer SDK syntax
        pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(current_index)
        
        # Create vector store using the index
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        
        retrieved_docs = vectorstore.similarity_search(state["question"])
        return {"context": retrieved_docs}
    except Exception as e:
        st.error(f"Error retrieving documents from index '{current_index}': {e}")
        return {"context": [Document(page_content=f"Error retrieving documents: {e}")]}


@traceable
def generate(state: State):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    prompt = hub.pull("response-generator")
    
    # Format context from retrieved documents
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Format chat history for context
    chat_history_text = ""
    if state["chat_history"]:
        for message in state["chat_history"]:
            role = message["role"]
            content = message["content"]
            chat_history_text += f"{role}: {content}\n"
    
    # Create messages with system prompt, chat history and current question
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
    ]
    
    # Add chat history if it exists
    if chat_history_text:
        messages.append(SystemMessage(content=f"Previous conversation:\n{chat_history_text}"))
    
    # Add context and question
    messages.append(SystemMessage(content=f"Context information:\n{docs_content}"))
    messages.append(HumanMessage(content=state["question"]))
    
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
    
    return {"answer": response.content}
 

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for selected index
if "selected_index" not in st.session_state:
    st.session_state.selected_index = DEFAULT_PINECONE_INDEX

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

# Sidebar with index selection
st.sidebar.title("Configuration")
available_indexes = get_available_indexes()
selected_index = st.sidebar.selectbox(
    "Select Pinecone Index",
    options=available_indexes,
    index=available_indexes.index(st.session_state.selected_index) if st.session_state.selected_index in available_indexes else 0
)

# Update session state if index changed
if selected_index != st.session_state.selected_index:
    st.session_state.selected_index = selected_index
    # Clear chat history when changing index
    st.session_state.messages = []
    st.sidebar.success(f"Switched to index: {selected_index}. Chat history cleared.")

# Add a separate button to clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.sidebar.success("Chat history cleared.")

st.image("https://www.projectpathways.org/assets/images/common/logo/logo-color.svg", width=100)
# Show title and description.
st.title("✨Segment Explorer AI Assistant✨")
st.write(
    "Ask me questions about Senegal Segmentation Data and the Pathways Methodology. I'll remember our conversation context."
)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
        
        # Format chat history for the state
        chat_history = [{"role": msg["role"], "content": msg["content"]} 
                        for msg in st.session_state.messages[:-1]]  # Exclude the current question
        
        # Invoke the graph with streaming
        result = graph.invoke({
            "question": prompt,
            "chat_history": chat_history,
            "context": [],
            "answer": None
        })
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
