# QueryLens 🧠

**QueryLens** is a multilingual Streamlit app that lets you ask questions from **webpages** or **PDF documents**. It uses LLM-based RAG (Retrieval-Augmented Generation) pipelines to give accurate answers directly from source content — in **any language** you ask.

---

## 🚀 Features

- 🌐 Ask from **Webpages** (via URL)
- 📄 Ask from **PDFs**
- 🌍 **Multilingual** support — ask in any language, get answers in the same language
- ⚡ Powered by **LLMs + LangChain** for smart, fast responses

---

## 🛠 How It Works

- `webloader(url)` → fetches and processes web content  
- `pdfloader(path)` → extracts and chunks PDF content  
- `qa_chain_web(docs)` / `qa_chain_pdf(docs)` → runs questions through an LLM  
- Uses **language-preserving** pipelines so answers match the query language

---

## 📦 Setup

```bash
git clone https://github.com/KushalJohari/QueryLens.git
cd QueryLens
pip install -r requirements.txt
streamlit run app.py
```

> Add your API keys (OpenAI, Groq, etc.) to a `.env` file if needed.

---

## 🧪 Example Use

- **URL Input:** `https://en.wikipedia.org/wiki/Artificial_intelligence`  
- **Question (Hindi):** "कृत्रिम बुद्धिमत्ता क्या है?"  
- **Answer:** In Hindi ✅

- **PDF Upload:** Academic paper  
- **Question (Spanish):** "¿Cuál es la conclusión del documento?"  
- **Answer:** In Spanish ✅

---

## 🗂 Folder Structure

```
querylens/
├── app.py           # Streamlit UI
├── Loaders.py       # Web/PDF loaders + QA chains
├── requirements.txt
└── README.md
```

---

## 📋 Requirements

```txt
streamlit
beautifulsoup4
langchain
langchain-groq
langchain-huggingface
langchain-community
```

---

## 👨‍💻 Author

**Kushal** — [GitHub](https://github.com/KushalJohari/QueryLens.git) | [LinkedIn](https://www.linkedin.com/in/kushal-johari-ba686527b/)

---
