# app/database/supabase_client.py
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

_supabase_client: Client = None

def get_supabase() -> Client:
    """Get Supabase client singleton."""
    global _supabase_client
    
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set in environment variables"
            )
        
        _supabase_client = create_client(url, key)
    
    return _supabase_client