"""
Configuration settings for the Boga Chat application.
"""
import os
from typing import List
from decouple import config, Csv


class Settings:
    """Application settings."""
    
    # API settings
    API_PREFIX: str = "/api"
    DEBUG: bool = config('DEBUG', default=False, cast=bool)
    
    # CORS settings
    CORS_ORIGINS: List[str] = config('CORS_ORIGINS', default="http://localhost:8501", cast=Csv())
    
    # Supabase settings
    SUPABASE_URL: str = config('SUPABASE_URL')
    SUPABASE_KEY: str = config('SUPABASE_KEY')
    
    # LangChain settings
    LANGCHAIN_API_KEY: str = config('LANGCHAIN_API_KEY')
    LANGCHAIN_PROJECT: str = config('LANGCHAIN_PROJECT', default="boga-chat")
    LANGCHAIN_TRACING_V2: bool = config('LANGCHAIN_TRACING_V2', default=True, cast=bool)
    LANGCHAIN_ENDPOINT: str = config('LANGCHAIN_ENDPOINT', default="https://api.smith.langchain.com")
    
    # LLM settings
    OPENAI_API_KEY: str = config('OPENAI_API_KEY')


# Create settings instance
settings = Settings() 