from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
import time
import textwrap
import gradio as gr
from langchain_groq import ChatGroq

loader = PyPDFDirectoryLoader("data")
the_text = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(the_text)

ollama_api_key = ""
vector_store = Chroma.from_documents(
    documents=docs,
    collection_name="ollama_embeds",
    embedding=OllamaEmbeddings(model="nomic-embed-text")
)

retriever = vector_store.as_retriever()

groq_api_key = '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'
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-8b-8192",
    temperature=0
)

raq_template = """Answer the question based only on the following context: {context}

Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(raq_template)

rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
)



def process_question(user_question):
    start_time = time.time()
    response = rag_chain.invoke(user_question)
    end_time = time.time()
    response_time = f"Response time: {end_time - start_time:.2f} seconds."
    full_response = f"{response}\n\n{response_time}"
    return full_response

iface = gr.Interface(fn=process_question,
                     inputs=gr.Textbox(lines=2, placeholder="Type your question here..."),
                     outputs=gr.Textbox(),
                     title="Personal Knowledge Chat App",
                     description="Ask any question about your document, and get an answer along with the response time."
                     )
iface.launch(share=True)
