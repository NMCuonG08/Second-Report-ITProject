## Embedding and Similarity search

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
import time
import textwrap
from langchain_groq import ChatGroq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
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

query = "Explain the concept of machine learning"

query_embedding = OllamaEmbeddings(model="nomic-embed-text").embed_query(query)
similar_docs = retriever.invoke(query)
for idx, doc in enumerate(similar_docs):
    doc_embedding = OllamaEmbeddings(model="nomic-embed-text").embed_documents([doc.page_content])[0]

    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
    print(f"Document {idx + 1}:\n{textwrap.fill(doc.page_content, width=100)}\n")
    print(f"Query Embedding: {query_embedding[:10]}... ")
    print(f"Document Embedding: {doc_embedding[:10]}...")
    print(f"Cosine Similarity: {similarity:.4f}")
    print("=" * 100)
```

Với đoạn code này em sẽ sử dụng Embedding model từ  [ollama](ollama.com) tên là nomic-embed-text








