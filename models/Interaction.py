
from typing import Any, Optional, List, Dict, Union
from pydantic import (
    BaseModel
)
# Pydantic models for request/response
class PromptRequest(BaseModel):
    text: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500


class ZeroShotRequest(BaseModel):
    task: str
    input_text: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500


class FewShotRequest(BaseModel):
    task: str
    examples: List[Dict[str, str]]  # [{"input": "...", "output": "..."}]
    input_text: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500


class ChainOfThoughtRequest(BaseModel):
    problem: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 800


class RoleBasedRequest(BaseModel):
    role: str
    task: str
    context: Optional[str] = ""
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500


class PromptResponse(BaseModel):
    response: str
    prompt_used: str
    tokens_used: int
    model: str
    timestamp: str

class ComparisonRequest(BaseModel):
    task: str
    input_text: str
    examples: Optional[List[Dict[str, str]]] = None
    role: Optional[str] = None


