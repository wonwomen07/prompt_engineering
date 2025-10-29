from typing import List, Dict


def generate_zero_shot_prompt(task: str, input_text: str) -> str:
    """Generate a zero-shot prompt for the model"""
    return f"""Task: {task}

Input: {input_text}

Please complete this task clearly and accurately."""


def generate_few_shot_prompt(task: str, input_text: str, examples: List[Dict[str, str]]) -> str:
    """Generate a few-shot prompt with examples"""
    examples_text = "\n".join([
        f"Input: {ex['input']}\nOutput: {ex['output']}\n"
        for ex in examples
    ])

    return f"""Task: {task}

Here are some examples:

{examples_text}

Now, please complete this task:
Input: {input_text}
Output:"""


def generate_chain_of_thought_prompt(problem: str) -> str:
    """Generate a chain-of-thought prompt for step-by-step reasoning"""
    return f"""Solve this problem step by step, showing your reasoning clearly:

Problem: {problem}

Please think through this step by step:
1. First, identify what we know
2. Then, determine what we need to find
3. Next, work through the solution methodically
4. Finally, state your answer clearly

Let's work through this together:"""


def generate_role_based_prompt(role: str, task: str, context: str = "") -> str:
    """Generate a role-based prompt with specific persona"""
    context_section = f"\nContext: {context}" if context else ""

    return f"""You are {role}.{context_section}

Task: {task}

Please respond from your expertise and perspective as {role}. Use your specialized knowledge and approach this task as a professional in this field would."""


def generate_template_prompt(text: str) -> str:
    """Generate a template-based prompt with structured output"""
    return f"""Please analyze the following text using this structured template:

INPUT TEXT: {text}

ANALYSIS TEMPLATE:
=================
SUMMARY: [2-3 sentence summary]

KEY POINTS:
• [Point 1]
• [Point 2] 
• [Point 3]

TONE & STYLE: [Description of writing style]

MAIN THEMES: [Primary themes identified]

RECOMMENDATIONS: [2-3 actionable suggestions]

TARGET AUDIENCE: [Who this is written for]

EFFECTIVENESS RATING: [1-10 with brief justification]

Please fill out each section of this template based on your analysis."""


def generate_advanced_prompt(text: str) -> str:
    """Generate an advanced prompt combining multiple techniques"""
    return f"""You are an expert content strategist and communication specialist with 10+ years of experience.

TASK: Comprehensive Content Analysis & Strategy

CONTENT TO ANALYZE:
{text}

INSTRUCTIONS:
Please follow this multi-step process:

STEP 1 - INITIAL ASSESSMENT
Think through what type of content this is and its apparent purpose.

STEP 2 - DETAILED ANALYSIS  
Analyze the content systematically across these dimensions:
- Clarity and readability
- Audience appropriateness  
- Persuasiveness and engagement
- Structure and organization
- Call-to-action effectiveness

STEP 3 - STRATEGIC RECOMMENDATIONS
Based on your analysis, provide specific, actionable recommendations.

STEP 4 - IMPLEMENTATION PRIORITY
Rank your recommendations by impact and ease of implementation.

OUTPUT FORMAT:
Present your analysis in a clear, professional report format suitable for stakeholders.

CONSTRAINTS:
- Be specific and actionable
- Support recommendations with reasoning
- Consider both short-term and long-term implications
- Maintain professional tone throughout

Begin your analysis:"""


# Specialized prompt generators
def generate_sentiment_prompts(text: str) -> Dict[str, str]:
    """Generate different sentiment analysis prompts"""
    return {
        "zero_shot": f"Analyze the sentiment of this text and classify it as positive, negative, or neutral:\n\n'{text}'",

        "few_shot": f"""Analyze sentiment and classify as positive, negative, or neutral:

Examples:
"I love this product!" → Positive
"This is terrible quality" → Negative  
"It's okay, nothing special" → Neutral
"Absolutely amazing experience!" → Positive
"Worst purchase ever" → Negative

Text: "{text}"
Sentiment:""",

        "chain_of_thought": f"""Analyze the sentiment of this text step by step:

Text: "{text}"

Step 1: Identify emotional words and phrases
Step 2: Consider overall tone and context
Step 3: Weigh positive vs negative elements
Step 4: Determine final sentiment classification

Let me work through this:""",

        "role_based": f"""You are an expert sentiment analysis specialist with years of experience in natural language processing.

Analyze this text and provide a professional sentiment assessment:
"{text}"

Please provide:
- Primary sentiment (positive/negative/neutral)
- Confidence level (1-10)
- Key indicators that led to this classification
- Any nuances or mixed sentiments detected"""
    }


def generate_summarization_prompts(text: str, technique: str, summary_length: str) -> str:
    """Generate text summarization prompts"""
    length_guides = {
        "short": "1-2 sentences",
        "medium": "3-4 sentences",
        "long": "a full paragraph"
    }

    prompts = {
        "zero_shot": f"Summarize this text in {length_guides[summary_length]}:\n\n{text}",

        "structured": f"""Please summarize the following text using this structure:

TEXT TO SUMMARIZE:
{text}

SUMMARY FORMAT:
- Main Point: [Core message in one sentence]
- Key Details: [2-3 supporting points]
- Conclusion: [Final takeaway]

Please provide a {summary_length} summary following this format.""",

        "chain_of_thought": f"""Summarize this text by thinking through it step by step:

Text: {text}

Step 1: Identify the main topic and purpose
Step 2: Extract key supporting points
Step 3: Note any important conclusions or outcomes
Step 4: Synthesize into a coherent {summary_length} summary

Let me work through this systematically:""",

        "role_based": f"""You are a professional editor and content strategist.

Create a {summary_length} summary of this text that captures the essential information while maintaining clarity and engagement:

{text}

As an expert, focus on what readers need to know most."""
    }

    return prompts.get(technique, prompts["zero_shot"])


def generate_content_creation_prompts(topic: str, content_type: str, technique: str, target_audience: str) -> str:
    """Generate content creation prompts"""
    prompts = {
        "zero_shot": f"Write a {content_type} about {topic} for {target_audience} audience.",

        "constraint_based": f"""Write a {content_type} about {topic} with these requirements:
- Target audience: {target_audience}
- Length: 200-300 words
- Tone: Professional yet engaging
- Include: Introduction, main points, conclusion
- Must include at least one actionable insight
- Use active voice
- End with a thought-provoking question""",

        "role_based": f"""You are an expert content creator specializing in {content_type} writing.

Create a compelling {content_type} about {topic} for {target_audience} audience.

Use your professional expertise to craft content that engages, informs, and provides value to the reader.""",

        "template_based": f"""Create a {content_type} about {topic} using this template:

HEADLINE: [Attention-grabbing title]

HOOK: [Opening that captures interest]

MAIN CONTENT:
• Point 1: [Key insight with example]
• Point 2: [Supporting information] 
• Point 3: [Actionable advice]

CONCLUSION: [Summary and call-to-action]

TARGET AUDIENCE: {target_audience}
Please fill in each section thoughtfully."""
    }

    return prompts.get(technique, prompts["zero_shot"])


def generate_code_generation_prompts(task: str, language: str, technique: str) -> str:
    """Generate code generation prompts"""
    prompts = {
        "zero_shot": f"Write {language} code to {task}.",

        "detailed_specification": f"""Write {language} code to {task}.

Requirements:
- Include proper error handling
- Add clear comments explaining the logic
- Follow best practices for {language}
- Include example usage
- Make the code modular and reusable

Please provide complete, working code.""",

        "step_by_step": f"""Write {language} code to {task} by following these steps:

Step 1: Plan the overall structure and approach
Step 2: Identify the main components needed
Step 3: Write the core logic with proper error handling
Step 4: Add helper functions if needed
Step 5: Include example usage and comments

Please show your thought process and then provide the complete code.""",

        "role_based": f"""You are a senior {language} developer with expertise in writing clean, efficient code.

Task: {task}

Please write production-quality {language} code that follows best practices, includes proper documentation, and demonstrates professional coding standards."""
    }

    return prompts.get(technique, prompts["zero_shot"])