## :rocket:  Embedding and Similarity search

With Model : `nomic-embed-text`

Use a multi-page PDF and a query related to multiple pages.

### 1. Using English documentation.

`query = "What is difference between Artificial Intelligence and machine learning and deep learning"`


 
![image](https://github.com/user-attachments/assets/8f6383e8-d761-4dae-a843-3f63866827aa)


The model's quality in this case with English seems quite good, as the cosine similarity between the query and the documents (PDF pages) is relatively high, specifically 0.8540 and 0.8520. This indicates that the model has successfully identified documents closely related to the query.

### 2. Using Vietnamese documentation.

`query = "Sự khác biệt giữa Trí tuệ nhân tạo và học máy và học sâu"`

![image](https://github.com/user-attachments/assets/0bcd7846-06b9-4dc6-86ba-20f6888f036c)



Accuracy of Cosine Similarity: The cosine similarity for the Vietnamese PDF pages is significantly lower than for the English pages. Specifically, the highest similarity value for Vietnamese pages is 0.6903, while for English, it reached 0.8540. This indicates that the model has more difficulty understanding and processing Vietnamese compared to English.

Quality of embeddings: The embeddings for both the queries and the Vietnamese documents show differences but are not as strong as in English. This may be because the model is not as well-trained on Vietnamese or lacks sufficient Vietnamese data for accurate processing.

Semantic understanding: Although there is some similarity between the query and the Vietnamese document content, the model seems to struggle more with understanding the context and complex details of the language. The differences in embeddings between the document and query are not as aligned, which could reduce the accuracy of the results.



## :rocket:  Model: mxbai-embed-large

### 1. Using English documentation.

![image](https://github.com/user-attachments/assets/c8ba6b35-cb02-424c-a207-678a091ad22a)

High results: The cosine similarity for the document with the corporate address is 0.8414, while the first document about the differences between AI, machine learning, and deep learning has a value of 0.8031. These values indicate that the model has identified a good level of similarity with the query

### 2. Using Vietnamese documentation.




## :rocket:  Model: jina/jina-embeddings-v2-base-en

### 1. Using English documentation.

![image](https://github.com/user-attachments/assets/a4079521-44a7-410d-bad9-339d3d2eb285)

High quality: These cosine similarity values indicate that jina/jina-embeddings-v2-base-en is very effective in determining the correlation between queries and documents, especially in technical fields such as AI.

### 2. Using Vietnamese documentation.

![image](https://github.com/user-attachments/assets/7d44ff4f-a996-4628-82d1-7825e32f5545)

These cosine similarity values indicate that the embedding models have effectively identified the correlation between the query and the content of the documents, enabling the provision of useful and relevant information to users. The use of Vietnamese does not diminish the quality of the content; rather, it helps preserve the meaning and context of the terms in the field of AI.







