from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from models.Interaction import PromptResponse, ZeroShotRequest, FewShotRequest,ChainOfThoughtRequest, RoleBasedRequest,PromptRequest, ComparisonRequest
from  model_interact.openai_interact import call_openai

from prompt_types.prompt import (generate_zero_shot_prompt,generate_few_shot_prompt,generate_chain_of_thought_prompt,generate_role_based_prompt,generate_template_prompt,generate_advanced_prompt,
                                 generate_summarization_prompts,
                                 generate_sentiment_prompts,
                                 generate_code_generation_prompts,
                                 generate_content_creation_prompts)
from services.prompt_service import PromptService
# Initialize services
prompt_service = PromptService()

# Initialize FastAPI app
app = FastAPI(
    title="Prompt Engineering API",
    description="Demonstrate different prompt engineering techniques using GPT-4o-mini",
    version="1.0.0"
)
#routes starts
#get
@app.get("/")
async def root():
    return {
        "message": "Prompt Engineering API with GPT-4o-mini",
        "endpoints": [
            "/zero-shot", "/few-shot", "/chain-of-thought",
            "/role-based", "/template-prompt", "/advanced-prompt",
            "/compare-techniques", "/docs"
        ]
    }



# zero shot prompt
@app.post("/zero-shot", response_model=PromptResponse)
async def zero_shot_prompting(request: ZeroShotRequest):
    """
    Zero-shot prompting: Ask the model to perform a task without examples
    """
    prompt = generate_zero_shot_prompt(request.task, request.input_text)

    result = await call_openai(prompt, request.temperature, request.max_tokens)

    return PromptResponse(
        response=result["response"],
        prompt_used=prompt,
        tokens_used=result["tokens_used"],
        model=result["model"],
        timestamp=datetime.now().isoformat()
    )

@app.post("/few-shot", response_model=PromptResponse)
async def few_shot_prompting(request: FewShotRequest):
    """
    Few-shot prompting: Provide examples to guide the model's response
    """
    prompt = generate_few_shot_prompt(
        task=request.task,
        input_text=request.input_text,
        examples=request.examples
    )

    result = await call_openai(prompt, request.temperature, request.max_tokens)

    return PromptResponse(
        response=result["response"],
        prompt_used=prompt,
        tokens_used=result["tokens_used"],
        model=result["model"],
        timestamp=datetime.now().isoformat()
    )


@app.post("/chain-of-thought", response_model=PromptResponse)
async def chain_of_thought_prompting(request: ChainOfThoughtRequest):
    """Chain-of-thought prompting: Encourage step-by-step reasoning"""
    prompt = generate_chain_of_thought_prompt(request.problem)
    result = await call_openai(prompt, request.temperature, request.max_tokens)

    return PromptResponse(
        response=result["response"],
        prompt_used=prompt,
        tokens_used=result["tokens_used"],
        model=result["model"],
        timestamp=datetime.now().isoformat()
    )


@app.post("/role-based", response_model=PromptResponse)
async def role_based_prompting(request: RoleBasedRequest):
    """Role-based prompting: Give the model a specific persona/expertise"""
    prompt = generate_role_based_prompt(request.role, request.task, request.context)
    result = await call_openai(prompt, request.temperature, request.max_tokens)

    return PromptResponse(
        response=result["response"],
        prompt_used=prompt,
        tokens_used=result["tokens_used"],
        model=result["model"],
        timestamp=datetime.now().isoformat()
    )


@app.post("/template-prompt", response_model=PromptResponse)
async def template_prompting(request: PromptRequest):
    """Template-based prompting: Structured format with clear sections"""
    prompt = generate_template_prompt(request.text)
    result = await call_openai(prompt, request.temperature, request.max_tokens)

    return PromptResponse(
        response=result["response"],
        prompt_used=prompt,
        tokens_used=result["tokens_used"],
        model=result["model"],
        timestamp=datetime.now().isoformat()
    )


@app.post("/advanced-prompt", response_model=PromptResponse)
async def advanced_prompting(request: PromptRequest):
    """Advanced prompting: Combines multiple techniques"""
    prompt = generate_advanced_prompt(request.text)
    result = await call_openai(prompt, request.temperature, request.max_tokens)

    return PromptResponse(
        response=result["response"],
        prompt_used=prompt,
        tokens_used=result["tokens_used"],
        model=result["model"],
        timestamp=datetime.now().isoformat()
    )


@app.post("/compare-techniques")
async def compare_prompting_techniques(request: ComparisonRequest):
    """Compare different prompting techniques on the same task"""
    return await prompt_service.compare_techniques(request)


@app.post("/sentiment-analysis")
async def sentiment_analysis_demo(text: str, technique: str = "zero_shot"):
    """Demonstrate sentiment analysis with different prompting techniques"""
    return await prompt_service.sentiment_analysis(text, technique)


@app.post("/text-summarization")
async def text_summarization_demo(text: str, technique: str = "zero_shot", summary_length: str = "medium"):
    """Demonstrate text summarization with different approaches"""
    return await prompt_service.text_summarization(text, technique, summary_length)


@app.post("/content-generation")
async def content_generation_demo(topic: str, content_type: str, technique: str = "zero_shot",
                                  target_audience: str = "general"):
    """Demonstrate content generation with various prompting approaches"""
    return await prompt_service.content_generation(topic, content_type, technique, target_audience)


@app.post("/code-generation")
async def code_generation_demo(task: str, language: str, technique: str = "zero_shot"):
    """Demonstrate code generation with different prompting techniques"""
    return await prompt_service.code_generation(task, language, technique)


@app.get("/templates")
async def get_prompt_templates():
    """Get predefined prompt templates for common tasks"""
    return prompt_service.get_templates()


@app.get("/examples/sentiment-analysis")
async def sentiment_examples():
    """Get example requests for sentiment analysis"""
    return prompt_service.get_sentiment_examples()


@app.get("/examples/text-summarization")
async def summarization_examples():
    """Get example requests for text summarization"""
    return prompt_service.get_summarization_examples()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "gpt-4o-mini", "timestamp": datetime.now().isoformat()}


# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)

"""

# 2. Zero-shot prompting
POST /zero-shot
{
    "task": "Translate to French",
    "input_text": "Hello, how are you today?"
}

# 3. Few-shot prompting  
POST /few-shot
{
    "task": "Extract key information",
    "examples": [
        {"input": "Great product, fast shipping!", "output": "Product: Positive, Shipping: Positive"},
        {"input": "Poor quality, slow delivery", "output": "Product: Negative, Shipping: Negative"}
    ],
    "input_text": "Amazing quality but took forever to arrive"
}

# 4. Chain-of-thought
POST /chain-of-thought
{
    "problem": "A store has 150 apples. They sell 60% in the morning and 25% of the remainder in the afternoon. How many apples are left?"
}

# 5. Role-based prompting
POST /role-based
{
    "role": "experienced marketing manager",
    "task": "Create a social media strategy for a new coffee shop",
    "context": "Located in a college town, targeting students and young professionals"
}

# 6. Compare techniques
POST /compare-techniques
{
    "task": "Analyze customer feedback",
    "input_text": "The app crashes frequently but has great features",
    "examples": [
        {"input": "Love the design, hate the bugs", "output": "Mixed: Positive design, Negative functionality"}
    ],
    "role": "product manager"
}

# 7. Sentiment analysis demo
POST /sentiment-analysis?technique=chain_of_thought
Body: "I'm really disappointed with this purchase, but the customer service was helpful"

# 8. Content generation
POST /content-generation
{
    "topic": "sustainable technology",
    "content_type": "blog post",
    "technique": "constraint_based",
    "target_audience": "business executives"
}
"""