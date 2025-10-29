from typing import List, Optional, Dict, Any
from openai import OpenAI
import os
from utils.libs import Libs
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

# Load .env file
load_dotenv()
utils= Libs()
utils.load_env()
# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))



# Helper function to call OpenAI API
async def call_openai(prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> Dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "response": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens,
            "model": "gpt-4o-mini"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
