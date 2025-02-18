# 📄 PDF-ChatBot  

A **Streamlit-based chatbot** that allows users to upload a **PDF file** and ask questions about its content using **Google Gemini AI**. This project leverages **LangChain, ChromaDB, and Google Generative AI** for document processing and **retrieval-augmented generation (RAG)**.  

## 🚀 Features  
✅ **Upload PDF Files** – Easily upload and process PDF documents  
✅ **Chunking & Embedding** – Extracts text, splits it, and stores embeddings using **ChromaDB**  
✅ **Conversational AI** – Ask questions and get relevant answers from **Google Gemini AI**  
✅ **Efficient Retrieval** – Uses **LangChain** to fetch the most relevant document chunks  
✅ **Fast & Interactive UI** – Built with **Streamlit** for an intuitive chatbot experience  

## 🛠️ Tech Stack  
- **Streamlit** – Frontend UI  
- **LangChain** – Document Processing & Retrieval  
- **ChromaDB** – Vector Storage for embeddings  
- **Google Gemini AI** – Large Language Model (LLM)  
- **Python** – Core programming language  

## 📌 Installation & Setup  

```bash
1️⃣ Clone the Repository  

git clone https://github.com/luxshan21/PDF-ChatBot.git
cd PDF-ChatBot

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Set Up Google API Key

GOOGLE_API_KEY=your_api_key_here

Or set it in your environment:

export GOOGLE_API_KEY=your_api_key_here

4️⃣ Run the Application

streamlit run pdf_chatbot_app.py
