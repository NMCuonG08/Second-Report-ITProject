## :rocket:  Embedding and Similarity search

With Model : `nomic-embed-text`

Use a `multi-page PDF` and a query related to multiple pages.

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

> When the discrepancy is large, it indicates that the model is not well-optimized for Vietnamese, resulting in weaker semantic representation of Vietnamese text compared to English.

## :rocket:  Model: mxbai-embed-large

### 1. Using English documentation.

![image](https://github.com/user-attachments/assets/c8ba6b35-cb02-424c-a207-678a091ad22a)

High results: The cosine similarity for the document with the corporate address is 0.8414, while the first document about the differences between AI, machine learning, and deep learning has a value of 0.8031. These values indicate that the model has identified a good level of similarity with the query

### 2. Using Vietnamese documentation.

![image](https://github.com/user-attachments/assets/5ea43c1f-6a1d-4764-8a2b-96794ad1f763)

 => The cosine similarity results still have relatively high values, the Vietnamese texts may still be relevant to the query, but their semantic representation might not be as accurate as in English.

## :rocket:  Model: jina/jina-embeddings-v2-base-en

### 1. Using English documentation.

![image](https://github.com/user-attachments/assets/a4079521-44a7-410d-bad9-339d3d2eb285)

High quality: These cosine similarity values indicate that jina/jina-embeddings-v2-base-en is very effective in determining the correlation between queries and documents, especially in technical fields such as AI.

### 2. Using Vietnamese documentation.

![image](https://github.com/user-attachments/assets/7d44ff4f-a996-4628-82d1-7825e32f5545)

These cosine similarity values indicate that the embedding models have effectively identified the correlation between the query and the content of the documents, enabling the provision of useful and relevant information to users. The use of Vietnamese does not diminish the quality of the content; rather, it helps preserve the meaning and context of the terms in the field of AI.

> A model performs well with multiple languages: When the discrepancy is small, it indicates that the model performs fairly well in both languages, with nearly equivalent semantic representation for both English and Vietnamese.

## :rocket:  Model: `Qwen2` is a new series of large language models from Alibaba

### 1. Using English documentation.

![image](https://github.com/user-attachments/assets/f231e34d-4abb-4097-823d-030acf141492)



### 2. Using Vietnamese documentation.


![image](https://github.com/user-attachments/assets/06e134b2-7a10-40e8-9409-224a10bda0e4)

> The performance of the "qwen2" embedding model, which appears to be more effective in processing Vietnamese compared to English, raises several considerations. Potential reasons for this may include differences in the training datasets used for each language, the model's architecture, or the specific characteristics of the languages themselves

---

- Comparison: In English document:` jina/jina-embeddings-v2-base-e`n > `nomic-embed-text` >  `mxbai-embed-large` > `qwen2`
  
 -  In Vietnamese document: `jina/jina-embeddings-v2-base-en` >` qwen2` > `nomic-embed-text` >  `mxbai-embed-large` 
---

## :rocket: Use multiple PDFs with related or similar content

With Model : `nomic-embed-text`

query = "`What is artificial intelligence`"

query = "`Trí tuệ nhân tạo là gì`"

1. Using English documentation.





![image](https://github.com/user-attachments/assets/e7cf84f4-b4d8-4f65-943a-7280bbbacc69)





2. Using Vietnamese documentation.


![image](https://github.com/user-attachments/assets/289fbc0c-689e-470b-b474-97037bb3ead4)



 ## qwen2

1. Using English documentation.

![image](https://github.com/user-attachments/assets/7155cfd0-c936-488a-80f4-ca06bdd02123)


![image](https://github.com/user-attachments/assets/677d9564-0f0f-4e01-ac7a-ee84cf647a75)

2. Using Vietnamese documentation.

![image](https://github.com/user-attachments/assets/2306e845-5672-47be-9186-57b6049f4b7f)




## mxbai-embed-large

1. Using English documentation.

![image](https://github.com/user-attachments/assets/84740b9d-afd1-43d3-8e31-a40db5d10641)

2. Using Vietnamese documentation.

![image](https://github.com/user-attachments/assets/cea706c5-dd20-41c7-8d03-c16cb833b259)


jina/jina-embeddings-v2-base-en

1. Using English documentation.

![image](https://github.com/user-attachments/assets/f60ecddc-54fc-4cd7-bb43-857e152b5186)


2. Using Vietnamese documentation.

![image](https://github.com/user-attachments/assets/bc9a51bf-81da-4f89-aa7f-4759750e4a51)








