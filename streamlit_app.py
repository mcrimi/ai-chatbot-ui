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

#Keys
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "pathways-ai-assistant"

#Constants
PINECONE_INDEX_NAME = "senegal-metrics-v1-1"
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


@traceable
def retrieve(state: State):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    retrieved_docs = vectorstore.similarity_search(state["question"])
    return {"context": retrieved_docs}


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
