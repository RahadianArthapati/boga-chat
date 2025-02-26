# Boga Chat

A chatbot application built with LangChain, LangSmith, and LangGraph.

## Features

- Interactive chat interface with Streamlit
- FastAPI backend for handling chat requests
- Supabase integration for storing conversations and vector embeddings
- LangChain for building the chat components
- LangGraph for creating conversational workflows
- LangSmith for tracing and monitoring

## Project Structure

```
boga_chat/
│
├── backend/                      # FastAPI application
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration settings
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       └── chat.py      # Chat endpoints
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   └── supabase.py      # Supabase connection
│   │   └── langchain/
│   │       ├── __init__.py
│   │       ├── chains.py        # LangChain components
│   │       └── graphs.py        # LangGraph workflow
│   ├── requirements.txt         # Backend dependencies
│   └── .env                     # Environment variables
│
├── frontend/                    # Streamlit application
│   ├── app.py                   # Main Streamlit app
│   ├── requirements.txt         # Frontend dependencies
│   ├── utils/
│   │   └── api.py               # Backend API client
│   └── .streamlit/
│       └── config.toml          # Streamlit configuration
│
├── setup_backend.sh             # Setup script for backend (Mac/Linux)
├── setup_frontend.sh            # Setup script for frontend (Mac/Linux)
├── setup.bat                    # Setup script for Windows
└── README.md                    # Basic documentation
```

## Setup

### Prerequisites

- Python 3.9+
- Supabase account
- OpenAI API key
- LangChain API key

### Easy Setup (Using Scripts)

#### On Mac/Linux:

1. Make the setup scripts executable:
   ```
   chmod +x setup_backend.sh setup_frontend.sh
   ```

2. Run the setup scripts:
   ```
   ./setup_backend.sh
   ./setup_frontend.sh
   ```

#### On Windows:

1. Run the setup batch file:
   ```
   setup.bat
   ```

### Manual Setup

#### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys and Supabase credentials

5. Run the backend server:
   ```
   uvicorn app.main:app --reload
   ```

#### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Supabase Setup

1. Create a new Supabase project
2. Set up the following tables:

```sql
-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY,
    messages JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create vector extension if not exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table for vector search
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Usage

1. Start the backend server:
   ```
   cd backend
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   uvicorn app.main:app --reload
   ```

2. Start the Streamlit frontend:
   ```
   cd frontend
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   streamlit run app.py
   ```

3. Open your browser to http://localhost:8501
4. Start chatting with Boga!

## License

MIT 