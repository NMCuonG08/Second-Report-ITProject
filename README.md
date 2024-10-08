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

- Với đoạn code này em sẽ sử dụng Embedding model từ  [ollama](https://ollama.com/blog/embedding-models) tên là nomic-embed-text

#### Với câu truy vẫn là `query = "Explain the concept of machine learning"`
- Và một đoạn văn như file pdf [Machine Learning](https://github.com/NMCuonG08/Second-Report-ITProject/blob/main/data/Machine%20learning%20-%20Wikipedia.pdf)

![image](https://github.com/user-attachments/assets/1c706548-8e0e-42ab-80bf-5317063348a5)

- Sử dụng công thức tính Cosine Similarity bằng thư viện của `scikit-learn`


 $\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \times \|\mathbf{B}\|}$


Trong đó:
-  `A`  là vector của câu truy vấn.
- ` B ` là vector của tài liệu.
- $\{\mathbf{A} \cdot \mathbf{B}}$ là tích vô hướng giữa hai vector.
- $\{\|\mathbf{A}\| \times \|\mathbf{B}\|}$ là độ dài của vector ` A ` và ` B `.


- Với Cosine Similarity: 0.6781 cho thấy có sự tương đồng cao giữa thông tin của tài liệu và của câu truy vẫn 

#### Khi sử dụng câu truy vấn bằng tiếng việt như sau ` query = "Giải thích khái niệm học máy"`


  ![image](https://github.com/user-attachments/assets/a75eb4d1-840d-487f-a492-c5ae384eae8d)

- Có thể nhận thấy với trường hợp sử dụng tiếng  việt giá trị `Cosine Similarity: 0.3678` khá là thấp khi  mà so sánh với khi sử dụng với tiếng anh. Việc này có thể là do cấu trúc của tiếng anh khác so với tiếng việt. Hoặc mô hình này được huấn luyện trên dữ liệu bằng tiếng anh là chủ yếu và nó sẽ tối ưu tốt hơn trên tiếng anh. 

