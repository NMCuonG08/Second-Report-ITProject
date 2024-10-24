from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load PDF từ thư mục
loader = PyPDFDirectoryLoader("data")
pdf_docs = loader.load()

# Chia mỗi trang thành một tài liệu riêng biệt
docs = []
for idx, doc in enumerate(pdf_docs):
    # Gắn metadata để lưu số trang cho từng tài liệu
    doc.metadata['page_number'] = idx + 1  # Lưu số trang vào metadata
    docs.append(doc)

# Kiểm tra số lượng trang
num_docs = len(docs)
print(f"PDF đã được chia thành {num_docs} trang (mỗi trang là một tài liệu).")

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# Khởi tạo Ollama API và Chroma Vector Store
ollama_api_key = ""
vector_store = Chroma.from_documents(
    documents=docs,
    collection_name="ollama_embeds",
    embedding=OllamaEmbeddings(model="qwen2")
)

retriever = vector_store.as_retriever()

# Truy vấn
query = "What is difference between Artificial Intelligence and machine learning and deep learning"

#query = "Sự khác biệt giữa Trí tuệ nhân tạo và học máy và học sâu"
query_embedding = OllamaEmbeddings(model="qwen2").embed_query(query)

# Tạo danh sách lưu trữ kết quả gồm số trang, nội dung, và similarity score
results = []

# Lặp qua tất cả các tài liệu (trang) và tính similarity score cho từng trang
for idx, doc in enumerate(docs):
    doc_embedding = OllamaEmbeddings(model="qwen2").embed_documents([doc.page_content])[0]

    # Tính toán cosine similarity giữa truy vấn và mỗi tài liệu
    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]

    # Lấy số trang từ metadata
    page_number = doc.metadata.get('page_number', 'Unknown')

    # Lưu kết quả vào danh sách `results`
    results.append({
        'page_number': page_number,
        'content': doc.page_content[:200],  # Hiển thị một phần nội dung
        'similarity': similarity,
        'doc_embedding': doc_embedding,
    })

# Sắp xếp danh sách `results` theo similarity score từ cao xuống thấp
results = sorted(results, key=lambda x: x['similarity'], reverse=True)

# Hiển thị kết quả đã sắp xếp
for result in results:
    print(f"Document (Page: {result['page_number']}):")
    print(f"Content: \n{textwrap.fill(result['content'], width=100)}...\n")
    print(f"Cosine Similarity: {result['similarity']:.4f}")
    print(f"Query Embedding: {query_embedding[:10]}... ")
    print(f"Document Embedding: {result['doc_embedding'][:10]}...")
    print("=" * 100)
