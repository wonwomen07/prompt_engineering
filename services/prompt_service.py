# ================================================================
# services/prompt_service.py
from fastapi import HTTPException
from datetime import datetime
from typing import Dict, Any
from models.Interaction import PromptResponse, ComparisonRequest
from model_interact.openai_interact import call_openai
from prompt_types.prompt import (generate_zero_shot_prompt,generate_few_shot_prompt,generate_chain_of_thought_prompt,generate_role_based_prompt,generate_template_prompt,generate_advanced_prompt,
                                 generate_summarization_prompts,
                                 generate_sentiment_prompts,
                                 generate_code_generation_prompts,
                                 generate_content_creation_prompts)


class PromptService:
    """Service class for handling different prompt engineering tasks"""

    async def compare_techniques(self, request: ComparisonRequest) -> Dict[str, Any]:
        """Compare different prompting techniques on the same task"""
        results = {}

        # Zero-shot
        zero_shot_prompt = generate_zero_shot_prompt(request.task, request.input_text)
        results["zero_shot"] = await call_openai(zero_shot_prompt)

        # Few-shot (if examples provided)
        if request.examples:
            few_shot_prompt = generate_few_shot_prompt(request.task, request.input_text, request.examples)
            results["few_shot"] = await call_openai(few_shot_prompt)

        # Chain-of-thought
        cot_prompt = generate_chain_of_thought_prompt(f"Task: {request.task}\nInput: {request.input_text}")
        results["chain_of_thought"] = await call_openai(cot_prompt)

        # Role-based (if role provided)
        if request.role:
            role_prompt = generate_role_based_prompt(request.role, f"{request.task}\nInput: {request.input_text}")
            results["role_based"] = await call_openai(role_prompt)

        return {
            "comparison_results": results,
            "task": request.task,
            "input": request.input_text,
            "timestamp": datetime.now().isoformat(),
            "techniques_compared": list(results.keys())
        }

    async def sentiment_analysis(self, text: str, technique: str) -> PromptResponse:
        """Perform sentiment analysis using specified technique"""
        techniques = generate_sentiment_prompts(text)

        if technique not in techniques:
            raise HTTPException(status_code=400, detail=f"Technique must be one of: {list(techniques.keys())}")

        result = await call_openai(techniques[technique])

        return PromptResponse(
            response=result["response"],
            prompt_used=techniques[technique],
            tokens_used=result["tokens_used"],
            model=result["model"],
            timestamp=datetime.now().isoformat()
        )

    async def text_summarization(self, text: str, technique: str, summary_length: str) -> PromptResponse:
        """Perform text summarization using specified technique"""
        prompt = generate_summarization_prompts(text, technique, summary_length)
        result = await call_openai(prompt)

        return PromptResponse(
            response=result["response"],
            prompt_used=prompt,
            tokens_used=result["tokens_used"],
            model=result["model"],
            timestamp=datetime.now().isoformat()
        )

    async def content_generation(self, topic: str, content_type: str, technique: str,
                                 target_audience: str) -> PromptResponse:
        """Generate content using specified technique"""
        prompt = generate_content_creation_prompts(topic, content_type, technique, target_audience)
        result = await call_openai(prompt)

        return PromptResponse(
            response=result["response"],
            prompt_used=prompt,
            tokens_used=result["tokens_used"],
            model=result["model"],
            timestamp=datetime.now().isoformat()
        )

    async def code_generation(self, task: str, language: str, technique: str) -> PromptResponse:
        """Generate code using specified technique"""
        prompt = generate_code_generation_prompts(task, language, technique)
        result = await call_openai(prompt, max_tokens=800)

        return PromptResponse(
            response=result["response"],
            prompt_used=prompt,
            tokens_used=result["tokens_used"],
            model=result["model"],
            timestamp=datetime.now().isoformat()
        )

    def get_templates(self) -> Dict[str, Any]:
        """Get predefined prompt templates"""
        return {
            "content_creation": {
                "zero_shot": "Write a {content_type} about {topic}.",
                "few_shot": "Write a {content_type} about {topic}. Here are some examples:\n{examples}\n\nNow write about: {input}",
                "role_based": "You are a professional {role}. Write a {content_type} about {topic} for {audience}."
            },
            "data_analysis": {
                "zero_shot": "Analyze this data and provide insights: {data}",
                "chain_of_thought": "Analyze this data step by step:\n1. First, examine the data structure\n2. Identify patterns and trends\n3. Draw meaningful conclusions\n4. Provide actionable recommendations\n\nData: {data}",
                "role_based": "You are a data scientist. Analyze this data and provide professional insights: {data}"
            },
            "text_classification": {
                "zero_shot": "Classify this text as {categories}: {text}",
                "few_shot": "Classify text into these categories: {categories}\n\nExamples:\n{examples}\n\nClassify: {text}",
                "template": "Text: {text}\nCategory: ___\nConfidence: ___\nReasoning: ___"
            }
        }

    def get_sentiment_examples(self) -> Dict[str, Any]:
        """Get example requests for sentiment analysis"""
        return {
            "zero_shot_example": {
                "text": "I absolutely love this new smartphone! The camera quality is incredible.",
                "technique": "zero_shot"
            },
            "few_shot_example": {
                "text": "The product is decent but overpriced for what you get.",
                "technique": "few_shot",
                "examples": [
                    {"input": "This is amazing!", "output": "Positive"},
                    {"input": "Terrible experience", "output": "Negative"},
                    {"input": "It's okay", "output": "Neutral"}
                ]
            }
        }

    def get_summarization_examples(self) -> Dict[str, Any]:
        """Get example requests for text summarization"""
        sample_text = """
        Artificial Intelligence has revolutionized numerous industries over the past decade. 
        From healthcare diagnostics to autonomous vehicles, AI systems are becoming increasingly 
        sophisticated and widespread. Machine learning algorithms can now process vast amounts 
        of data to identify patterns that would be impossible for humans to detect manually. 
        However, this rapid advancement also raises important questions about job displacement, 
        privacy, and the ethical use of AI technology. Companies and governments worldwide are 
        working to establish frameworks that promote innovation while protecting individual rights 
        and ensuring responsible AI development.
        """

        return {
            "sample_text": sample_text,
            "zero_shot_example": {
                "text": sample_text,
                "technique": "zero_shot",
                "summary_length": "medium"
            },
            "structured_example": {
                "text": sample_text,
                "technique": "structured",
                "summary_length": "medium"
            }
        }
