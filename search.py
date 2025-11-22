import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks  import StreamlitCallbackHandler 
import os 
from dotenv import load_dotenv

from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


arx_wr = ArxivAPIWrapper()
arx_tool = ArxivQueryRun(api_wrapper=arx_wr)
wiki_wr = WikipediaAPIWrapper()
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wr)
search = DuckDuckGoSearchRun(name="Search")

load_dotenv() #forgot this ***IMPORTANT**

os.environ["HFT"]=os.getenv("HF_TOKEN")
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
loader = WebBaseLoader('https://www.investopedia.com/personal-finance/top-highest-paying-jobs/')
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents,embedding).as_retriever()
ret_tool = create_retriever_tool(vectordb,"retrivtool" ,"search the web link" )



st.title("Search Engine")
st.sidebar.title("settings")
api = st.sidebar.text_input("Enter your groq api :", type = "password")

if api:
 llm1 = ChatGroq(groq_api_key=api, model_name="openai/gpt-oss-20b")
 if "text" not in st.session_state:
    st.session_state["text"] = [{"role":"assistant","content":"I am a assistant to search relevant websites for ur query"}]

 for text in st.session_state.text:
    st.chat_message(text["role"]).write(text["content"])

 if prompt:=st.chat_input(placeholder="enter ur query"):
    st.session_state.text.append({"role":"user","content":prompt})  # "content":prompt not "prompt"
    st.chat_message("user").write(prompt)

    llm1 = ChatGroq(groq_api_key=api, model_name="meta-llama/llama-4-maverick-17b-128e-instruct" , streaming=True) # gpt oss model automatically calls tool so whith those models it wornt run !!!
   #   Model Behavior: Some language models, especially smaller or open-source ones (like certain gpt-oss or Llama variants), might occasionally "misbehave" and attempt to use tools if tools were defined in the request, even when instructed not to via the tool_choice="none" parameter. so the error : "Tool choice is none, but model called a tool" occured
    tools = [wiki_tool,arx_tool,search,ret_tool]
    search_agent = initialize_agent(tools, llm1, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, tool_choice="search", handle_parsing_errors=True)   #handle_parsing_errors=True not handling_parsing-erros

    with st.chat_message("assistant"):
       cb= StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
       response=search_agent.run(st.session_state.text, callbacks=[cb])
       st.session_state.text.append({"role":"assistant", "content":response})
       st.write(response)