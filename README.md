# 📄 DocumentAnswerer 🤖🗂️

[![Hugging Face Spaces](https://img.shields.io/badge/Gradio%20App-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/Prince1010/DocumentAnswer)

DocumentAnswerer is an AI-powered web application that allows users to upload a document (PDF) and interact with it through natural language queries. It also supports **appointment booking via prompt-based interactions** like “call me”, “appointment”, or “book”.

---

## 🚀 Features

- 📄 Upload a PDF and chat with it
- 💬 Asks questions and gets context-aware answers
- 📅 Books appointments via prompts
- ✅ Email and phone validation for booking
- 📧 Email confirmation of booked appointments
- 🔄 Continue chatting even after appointment booking

---

## 🧠 How It Works

### 1. **Gradio Frontend**
Built using [Gradio](https://www.gradio.app/) for a simple and interactive UI — fully supported by HuggingFace Spaces.

### 2. **PDF Ingestion & Embedding**
- PDF uploaded via `handle_pdf_upload()`
- Text extracted using `PyMuPDF`
- Text chunked via `RecursiveCharacterTextSplitter`
- Chunks embedded with `all-MiniLM-L6-v2` model
- Stored using `FAISS` for semantic similarity search

### 3. **Query Answering (RAG Pipeline)**
- Query embedded using same model
- Top-k similar chunks retrieved from FAISS
- Passed to LLM (`Mistral-7B-Instruct-v0.1`) for final answer generation

### 4. **Intent Detection & Appointment Booking**
- Prompts like “book”, “appointment” trigger booking flow
- Collects Name, Email, Phone, Preferred Date
- Validates formats, sends confirmation email

---

## 🧰 Technologies Used

| Tool | Purpose |
|------|---------|
| **Gradio** | User Interface |
| **LangChain** | RAG pipeline, embeddings, intent handling |
| **FAISS** | Semantic search |
| **PyMuPDF** | PDF text extraction |
| **SentenceTransformers** | Text embeddings |
| **Mistral-7B-Instruct** | Answer generation |
| **Regex + Validation** | Email & phone verification |

---

## 🌐 Try It Out

🔗 [Live on HuggingFace Spaces](https://huggingface.co/spaces/Prince1010/DocumentAnswer)

---

## ⚠️ Limitations

- Only supports **plain text** (no tables/images/OCR)
- Processes **one PDF at a time**
- Potential latency due to external **API calls**

---

## 🔐 Security Note

To protect private data, consider running LLMs locally using `llama-cpp-python` with LangChain. API usage is easier for deployment, but local models ensure better privacy.

---

## 💡 Future Improvements

- ✅ OCR Support for tables and images
- 📁 Multi-document Q&A
- 🚀 Speed optimization via local LLM integration

---

## 📬 Contact

For feedback or queries, feel free to reach out or fork and contribute to this project!

---

Made with ❤️ using Gradio, LangChain, and HuggingFace!
