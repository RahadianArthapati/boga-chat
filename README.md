# Boga Chat

A chatbot application powered by LangChain, LangSmith, LangGraph, and Supabase with document retrieval capabilities.

## Features

- 💬 Chat with an AI assistant powered by OpenAI
- 📚 Upload and manage documents
- 🔍 Search documents using vector similarity
- 🧠 RAG (Retrieval Augmented Generation) for document-enhanced responses
- 📊 LangSmith tracing for monitoring and debugging
- 🚀 FastAPI backend with streaming support
- 📱 Streamlit frontend with a modern UI

## Setup

### Prerequisites

- Python 3.9+
- Supabase account
- OpenAI API key
- LangSmith API key (optional, for tracing)

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/boga-chat.git
   cd boga-chat
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the `backend` directory with your API keys:
   ```
   CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
   
   # Supabase
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   SUPABASE_CONNECTION_STRING=your_supabase_connection_string
   
   # LangChain
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY=your_langchain_api_key
   LANGCHAIN_PROJECT=boga-chat
   
   # OpenAI
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Set up Supabase tables and functions:
   ```bash
   python setup_supabase.py
   ```

5. Start the backend server:
   ```bash
   uvicorn app.main:app --reload --port 8080
   ```

### Frontend Setup

1. Create a virtual environment and install dependencies:
   ```bash
   cd frontend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Using RAG (Retrieval Augmented Generation)

RAG enhances the chatbot's responses by retrieving relevant documents from your knowledge base.

### Step 1: Upload Documents

1. Go to the Documents page by clicking "Go to Documents" in the sidebar
2. Upload text files (.txt, .md, .csv, .json) with the document content
3. Add metadata like title, author, source, and tags

### Step 2: Enable RAG in Chat

1. Go back to the Chat page
2. Toggle "Enable Document Retrieval (RAG)" in the sidebar
3. Ask questions related to your documents
4. The chatbot will retrieve relevant documents and use them to enhance its responses
5. You can view the retrieved documents by expanding the "Retrieved Documents" section

## Architecture

- **Backend**: FastAPI application with LangChain for AI processing
  - `app/api/routes/chat.py`: Chat API endpoints
  - `app/api/routes/documents.py`: Document API endpoints
  - `app/langchain/chains.py`: LangChain chat chains
  - `app/langchain/embeddings.py`: Document embeddings
  - `app/langchain/rag.py`: RAG implementation
  - `app/db/supabase.py`: Supabase client

- **Frontend**: Streamlit application
  - `app.py`: Main chat interface
  - `pages/documents.py`: Document management interface
  - `utils/api.py`: API client for backend communication

- **Database**: Supabase with pgvector for vector similarity search
  - `conversations`: Stores chat conversations
  - `document_embeddings`: Stores document chunks and embeddings

## Troubleshooting

### Document Retrieval Not Working

If the chatbot doesn't retrieve documents when RAG is enabled:

1. Make sure you've uploaded documents in the Documents page
2. Check that the pgvector extension is enabled in Supabase
3. Verify that the SQL functions are properly set up by running `setup_supabase.py`
4. Check the backend logs for any errors related to document retrieval

### API Connection Issues

If the frontend can't connect to the backend:

1. Make sure the backend server is running on port 8080
2. Check that the CORS settings in `backend/.env` include your frontend URL
3. Verify that the API client in the frontend is using the correct base URL

## License

MIT 