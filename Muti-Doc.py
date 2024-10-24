from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# Load PDF từ thư mục
loader = PyPDFDirectoryLoader("data")
pdf_docs = loader.load()

# Tạo danh sách các tài liệu
docs = []

# Xử lý từng tài liệu (file PDF) riêng biệt
for doc in pdf_docs:
    # Lấy metadata để lưu số trang gốc và tên tài liệu
    if 'page_number' not in doc.metadata:
        doc.metadata['page_number'] = doc.metadata.get('page', 'Unknown')  # Dùng metadata page từ tài liệu gốc
    doc.metadata['file_name'] = doc.metadata.get('source', 'Unknown')  # Lưu tên file gốc vào metadata
    docs.append(doc)

# Kiểm tra số lượng tài liệu đã tải
num_docs = len(docs)
print(f"PDF đã được chia thành {num_docs} trang (mỗi trang là một tài liệu).")

# Khởi tạo Ollama API và Chroma Vector Store
ollama_api_key = ""
vector_store = Chroma.from_documents(
    documents=docs,
    collection_name="nomic-embed-text",
    embedding=OllamaEmbeddings(model="nomic-embed-text")
)

retriever = vector_store.as_retriever()

# Truy vấn
query = "what is artificial intelligence?"
#query = "Trí tuệ nhân tạo là gì?"
query_embedding = OllamaEmbeddings(model="nomic-embed-text").embed_query(query)

# Tạo danh sách lưu trữ kết quả gồm số trang, tên tài liệu, nội dung, và similarity score
results = []

# Lặp qua tất cả các tài liệu (trang) và tính similarity score cho từng trang
for idx, doc in enumerate(docs):
    doc_embedding = OllamaEmbeddings(model="nomic-embed-text").embed_documents([doc.page_content])[0]

    # Tính toán cosine similarity giữa truy vấn và mỗi tài liệu
    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]

    # Lấy số trang và tên tài liệu từ metadata
    page_number = doc.metadata.get('page_number', 'Unknown')
    file_name = doc.metadata.get('file_name', 'Unknown')

    # Lưu kết quả vào danh sách results
    results.append({
        'page_number': page_number,
        'file_name': file_name,
        'content': doc.page_content[:50],  # Hiển thị một phần nội dung
        'similarity': similarity,
        'doc_embedding': doc_embedding,
    })

# Sắp xếp danh sách results theo similarity score từ cao xuống thấp
results = sorted(results, key=lambda x: x['similarity'], reverse=True)

# Hiển thị kết quả đã sắp xếp
current_page = None
current_file = None
for result in results:
    # Kiểm tra nếu tài liệu hoặc trang hiện tại khác, in ra phân cách và thông tin mới
    if result['file_name'] != current_file or result['page_number'] != current_page:
        current_file = result['file_name']
        current_page = result['page_number']
        print("\n" + "=" * 50)
        print(f"Tài liệu: {current_file} (Trang: {current_page})")
        print("=" * 50)

    # In nội dung và cosine similarity
    print(f"Nội dung: \n{textwrap.fill(result['content'], width=100)}...\n")
    print(f"Cosine Similarity: {result['similarity']:.4f}")
    print(f"Query Embedding: {query_embedding[:10]}... ")
    print(f"Document Embedding: {result['doc_embedding'][:10]}...")
    print("-" * 100)
