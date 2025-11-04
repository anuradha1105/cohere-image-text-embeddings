# 🧠 Cohere Image–Text Embeddings (embed-v4.0)

This project demonstrates **multimodal embeddings** using Cohere’s `embed-v4.0` model to generating vector representations for both **images** and **text queries** to measure semantic similarity.

## 🎯 Objective
Use Cohere’s `embed-v4.0` model to:
- Create embeddings for two SJSU images.
- Create embeddings for text queries:
   >- person with tape and cap
   >- cart with single tire
- Compare image: image and query : image pairs using **cosine similarity**.

## 🧩 Files in this repository
cohereembeddingreleasewithsearch.py - Main script 
similarities.csv - Output table containing cosine similarity values 
cohrere-image-text-colab.ipynb - Google Colab notebook with code run

---

## 🧪 Output Evidence

### 1️⃣ Console Output
Below is the screenshot of the terminal output after running  

<img width="1211" height="286" alt="image" src="https://github.com/user-attachments/assets/753a0e06-25ab-4228-ac4a-8ab075a91fd3" />


---

### 2️⃣ CSV Results Table
Below is the preview of `similarities.csv` displayed using Pandas:

<img width="662" height="189" alt="image" src="https://github.com/user-attachments/assets/7eb2b9a2-c3a5-4720-bd5f-220f90b8ab51" />


---

## 📊 Observations
| Query | Best Match Image | Cosine Similarity (Highest) | Interpretation |
|:--|:--|:--:|:--|
| person with tape and cap | ADV_college-of-science_2.jpg | 0.178 | People using tape and caps → science context |
| cart with single tire | ADV_college-of-social-sciences_2.jpg | 0.234 | Wheelbarrow/cart visible → social sciences image |

> The **“person with tape and cap”** query is more semantically similar to the *College of Science* image,  
> and the **“cart with single tire”** query aligns with the *College of Social Sciences* image.  
> This confirms that Cohere’s `embed-v4.0` model effectively captures shared meaning between text and image modalities.

---

## ⚙️ How to Run
1. Install dependencies:
   ```bash
   pip install cohere requests numpy pandas


## Set your API key:
```
export COHERE_API_KEY="your_api_key_here"
```

### Run the script:
```
python cohereembeddingreleasewithsearch.py
```

The results will print in the terminal and save to similarities.csv.

## 🏁 Summary
This experiment demonstrates how Cohere’s multimodal embeddings can align text and image concepts in the same semantic space.
By comparing embeddings with cosine similarity, we can identify which images best match specific text descriptions.
