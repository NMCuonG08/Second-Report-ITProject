## :rocket:  Embedding and Similarity search

With Model : `nomic-embed-text`

Use a multi-page PDF and a query related to multiple pages.

### 1. Using English documentation.

`query = "What is difference between Artificial Intelligence and machine learning and deep learning"`


 
![image](https://github.com/user-attachments/assets/8f6383e8-d761-4dae-a843-3f63866827aa)


The model's quality in this case with English seems quite good, as the cosine similarity between the query and the documents (PDF pages) is relatively high, specifically 0.8540 and 0.8520. This indicates that the model has successfully identified documents closely related to the query.

### 1. Using Vietnamese documentation.

`query = "Sự khác biệt giữa Trí tuệ nhân tạo và học máy và học sâu"`

![image](https://github.com/user-attachments/assets/0bcd7846-06b9-4dc6-86ba-20f6888f036c)



Accuracy of Cosine Similarity: The cosine similarity for the Vietnamese PDF pages is significantly lower than for the English pages. Specifically, the highest similarity value for Vietnamese pages is 0.6903, while for English, it reached 0.8540. This indicates that the model has more difficulty understanding and processing Vietnamese compared to English.

Quality of embeddings: The embeddings for both the queries and the Vietnamese documents show differences but are not as strong as in English. This may be because the model is not as well-trained on Vietnamese or lacks sufficient Vietnamese data for accurate processing.

Semantic understanding: Although there is some similarity between the query and the Vietnamese document content, the model seems to struggle more with understanding the context and complex details of the language. The differences in embeddings between the document and query are not as aligned, which could reduce the accuracy of the results.

