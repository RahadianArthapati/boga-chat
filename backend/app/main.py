"""
Main FastAPI application entry point for Boga Chat.
"""
import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Configure LangChain tracing via environment variables
if settings.LANGCHAIN_TRACING_V2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    logging.info(f"LangChain tracing enabled for project: {settings.LANGCHAIN_PROJECT}")

app = FastAPI(
    title="Boga Chat API",
    description="API for Boga Chat, a chatbot powered by LangChain, LangSmith, and LangGraph",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Welcome to Boga Chat API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"} 