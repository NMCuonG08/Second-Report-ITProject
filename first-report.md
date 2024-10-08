# Second Report IT Project
about `Embedding` and `Similarity Search`



## 1. What is Embedding
![image](https://github.com/user-attachments/assets/fdaa04ae-198b-4185-aa5f-76f0fe324f57)
- Embedding is a representation of objects like text, images and audio as points in a continuous vector space.
- Word Embedding is a type of representation used in natural language processing (NLP) to map words in a numerical vectors

![image](https://github.com/user-attachments/assets/66adb212-fdc8-4107-85e9-1a10e48b2919)

- The distance between these vectors indicates the level of semantic similarity or difference between words.
- “man” and “woman” are close to each other and lie on the same line, indicating semantic similarity but a difference in gender.
- “king” and “queen” are also close, representing similar meanings (both are royalty) but differ in gender.

## 2. There are two main methods to calculate word embeddings: `Count-based` and `Prediction method`

### 2.1 Count-based

>  Count-based methods rely on counting the occurrences of words or pairs of words in a text .
>  Common techniques include Bag-of-Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), and Co-occurrence Matrix

Consider the following three sentences:
  
  - "The cat sat on the mat."
  - "The cat ate the fish."
  - "The dog sat on the mat."

|     | the | cat | sat | on | mat | ate | fish | dog |
|-----|-----|-----|-----|----|-----|-----|------|-----|
| the|  0  |  2  |  2  | 2  |  1  |  1  |  1   |  1  |
| cat | 2  |  0  |  1  | 0  |  0  |  1  |  1   |  0  |
| sat |  2  |  1  |  0  | 2  |  1  |  0  |  0   |  1  |
| on  |  2  |  0  |  2  | 0  |  1  |  0  |  0   |  1  |
| mat |  1  |  0  |  1  | 1  |  0  |  0  |  0   |  1  |
| ate |  1  |  1  |  0  | 0  |  0  |  0  |  1   |  0  |
| fish|  1  |  1  |  0  | 0  |  0  |  1  |  0   |  0  |
| dog |  1  |  0  |  1  | 1  |  1  |  0  |  0   |  0  |


- Advantages: Easy to compute and does not require large amounts of data.

- Disadvantages: The matrix size can be very large, lacks the ability to capture complex contextual relationships, and does not retain deep semantic relationships.

### 2.2 Prediction-based Methods

> Prediction-based methods (also known as prediction techniques) are used to learn word representations based on the ability to predict the context of a given word or vice versa.

> Popular models include Word2Vec (with two architectures: CBOW and Skip-gram), GloVe, and FastText.


#### Word2Vec (Skip-gram):

Consider the sentence: “The cat sat on the mat.”

  - - Skip-gram aims to predict the context words of a given target word. For example, for the target word “cat,” the context words might be {“The”, “sat”, “on”}. The model learns to adjust vectors so that the           word     “cat” and its context words have similar vectors.

#### Word2Vec (CBOW - Continuous Bag of Words):

  - - Instead of predicting context from the target word like Skip-gram, CBOW uses the context to predict the target word. For example, with the context {“The”, “sat”}, CBOW will try to predict the target word “cat.”

![image](https://github.com/user-attachments/assets/e17a7801-2c0d-4934-8613-92b7b1b1dc90)

## 2. What is Similarity Search

![image](https://github.com/user-attachments/assets/fb04156f-faad-4a7a-a441-33468f8a793a)


>Similarity Search uses vectors from Word Embeddings to measure the similarity between words or text segments using methods such as Cosine Similarity, Euclidean Distance, etc. This helps identify words with similar meanings or related content

- $\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \times \|\mathbf{B}\|}$


![image](https://github.com/user-attachments/assets/748ba2e2-5fe1-4b2e-b98c-7159080309ed)

- $\text{Euclidean Distance} = \sqrt{(A_1 - B_1)^2 + (A_2 - B_2)^2 + \ldots + (A_n - B_n)^2}$

![image](https://github.com/user-attachments/assets/d3306b76-48fe-4bbb-9f8d-132c7bddaa0c)



## 3. What is Retrieval Augmented Generation (RAG)

with the document 

![image](https://github.com/user-attachments/assets/236a3977-c9d5-4acc-9c66-a0ecc1b6667b)

After Chat app using RAG :

![image](https://github.com/user-attachments/assets/07c557c1-4bf6-48af-b076-1e0d2eb66b04)













