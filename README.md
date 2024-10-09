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

- Với đoạn code này em sẽ sử dụng Embedding model từ  [ollama](https://ollama.com/blog/embedding-models) tên là  [nomic-embed-text](https://ollama.com/library/nomic-embed-text)

#### Với câu truy vẫn là `query = "Explain the concept of machine learning"`
- Và một đoạn văn như file pdf [Machine Learning](https://github.com/NMCuonG08/Second-Report-ITProject/blob/main/data/Machine%20learning%20-%20Wikipedia.pdf)

![image](https://github.com/user-attachments/assets/1c706548-8e0e-42ab-80bf-5317063348a5)

- Document embedding là một cách biểu diễn nội dung của một tài liệu (document) thành một vector (mảng số) trong không gian nhiều chiều. Mỗi vector đại diện cho các đặc trưng của tài liệu, giúp mô hình có thể hiểu và xử lý nội dung của nó một cách hiệu quả.

- Query embedding là vector biểu diễn cho một truy vấn (query) mà người dùng đưa ra, trong trường hợp này là câu hỏi hoặc nội dung mà người dùng muốn tìm kiếm. Nó cũng là một vector trong không gian nhiều chiều, tương tự như document embedding.

- Cosine similarity là một phương pháp đo lường độ tương đồng giữa hai vector . Có giá trị từ -1 đến 1( tăng từ hoàn toàn trái ngược đến hoàn toàn tương đồng ).  Sử dụng công thức tính Cosine Similarity bằng thư viện của `scikit-learn`


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


## Với model thứ 2 em sử dụng model tên là [mxbai-embed-large](https://ollama.com/library/mxbai-embed-large)

![image](https://github.com/user-attachments/assets/f048df07-c554-4926-acfc-82283de76880)


![image](https://github.com/user-attachments/assets/7654357c-f190-4bab-b8f7-3e52f601d95a)

![image](https://github.com/user-attachments/assets/4e712238-77d3-446c-baae-f0652831fcf9)


- Với mô hình này thì `Cosine Similarity: 0.4535` với câu truy vấn tiếng việt và `Cosine Similarity: 0.729` khi câu truy vấn là tiếng anh. Có thể nhận thấy model này có vẻ chất lượng hơn khi có độ tương đồng cao hơn [nomic-embed-text](https://ollama.com/library/nomic-embed-text)


### Với model thứ 2 em sử dụng model tên là [jina/jina-embeddings-v2-base-en]([https://ollama.com/library/mxbai-embed-large](https://ollama.com/jina/jina-embeddings-v2-base-en))

![image](https://github.com/user-attachments/assets/a7f80fed-d319-471e-a3bd-ac625ff02bdd)

![image](https://github.com/user-attachments/assets/e036aff0-ae45-49b0-b65f-57b76f070742)


- Với mô hình này thì `Cosine Similarity: 0.6406` với câu truy vấn tiếng việt và `Cosine Similarity: 0.8909` khi câu truy vấn là tiếng anh. Có thể nhận thấy model này có chất lượng cao nhất khi so với 2 model kia.


## What is RAG ( Retrieval-Augmented Generation )
- Đây là một phương pháp trong lĩnh vực trí tuệ nhân tạo và xử lý ngôn ngữ tự nhiên (NLP) kết hợp giữa việc truy xuất thông tin và sinh văn bản. Phương pháp này thường được sử dụng để cải thiện khả năng tạo ra văn bản chất lượng cao hơn, bằng cách tích hợp thông tin từ một cơ sở dữ liệu hoặc tài liệu bên ngoài.

- Như em hiểu được thì chương trình sẽ tìm được các vector embeddings có các độ tương đồng với câu truy vấn và RAG sẽ tận dụng các embedding vector để xác định các thông tin liên quan và từ đó tạo ra các câu trả lời hợp lý dựa trên nội dung của tài liệu đã tìm thấy. Hệ thống  sử dụng từ khóa và ý tưởng chính từ các tài liệu này để sinh ra phản hồi phù hợp nhất với yêu cầu của người dùng.

```python
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
```

![image](https://github.com/user-attachments/assets/ce9e061c-5ed8-4750-8579-7f20f894a5d6)


- Giống như trên nó sẽ sinh ra đoạn văn bản từ đó.
 


