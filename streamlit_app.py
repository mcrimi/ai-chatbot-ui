import streamlit as st
from openai import OpenAI
import os
from langchain.chat_models import ChatOpenAI
from langchain import hub
from langchain.docstore.document import Document
from langsmith import traceable
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

#Keys
os.environ["PINECONE_API_KEY"] = "pcsk_4E5oQG_A8ZynYUqYSEwBeM6xghNRBQV7685AZ54JdvmC1oCafdeUmFAbBBLoqEDWcMGLKW"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "pathways-ai-assistant"

#Constants
PINECONE_INDEX_NAME = "senegal-metrics-v1-1"
EMBEDDING_MODEL = "text-embedding-3-large"


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

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
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.5, openai_api_key=OPENAI_API_KEY)
    response = llm.invoke(messages)
    return {"answer": response.content}

st.set_page_config(
    page_title="Segment Explorer AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': 'https://github.com/streamlit/streamlit/issues',
        'About': "### Segment Explorer AI Assistant - Powered by Streamlit"
    }
)

# Custom CSS for additional styling
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;pi
            color: #FAFAFA;
        }
        .stButton>button {
            color: white;
            background-color: #1E90FF;
        }
    </style>
""", unsafe_allow_html=True)

st.image("https://www.projectpathways.org/assets/images/common/logo/logo-color.svg", width=100)
# Show title and description.
st.title("✨Hi, I'm a very rudimentary version of the Segment Explorer AI Assistant✨")
st.write(
    "This is a simple demo of a conversational AI assistant that can answer questions based on a the Senegal Segmentation Data and the Pathways Methodology"
)

with st.form("my_form"):
    text = st.text_area(
        "Enter your question:"
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        result = graph.invoke({"question": text})
        answer = result["answer"]
        st.info(answer)
