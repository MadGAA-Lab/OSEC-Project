"""
Patient Agent - Simulates patient with personality-driven behavior
"""

import logging
import random
import time
from typing import List

from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionAssistantMessageParam

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Diverse fallback messages when API fails - natural patient responses
FALLBACK_MESSAGES = [
    "Sorry, what were you saying? I zoned out for a second there.",
    "Wait, can you repeat that? I'm having trouble focusing right now.",
    "I... I'm not sure what to say to that.",
    "Hold on, I need to think about this for a moment.",
    "I'm sorry, my mind is just racing right now.",
    "Can we slow down? This is a lot to process.",
    "I don't know... I'm really confused about all this.",
    "Everything you're saying is just... it's overwhelming.",
]


class PatientAgent:
    """
    Simulates patient with personality-driven behavior
    
    Uses full system prompt (MBTI personality + background + concerns)
    that is HIDDEN from Doctor Agent
    """
    
    def __init__(self, client: OpenAI, model: str, system_prompt: str, max_retries: int = MAX_RETRIES, retry_delay: int = RETRY_DELAY):
        """
        Initialize PatientAgent
        
        Args:
            client: OpenAI client for LLM calls
            model: Model name to use
            system_prompt: Full patient persona system prompt (includes personality)
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dialogue_history: list[dict] = []  # List of {role: str, content: str}
        
        logger.info(f"PatientAgent initialized with personality-driven system prompt (retries={max_retries}, delay={retry_delay}s)")
    
    def reset(self):
        """Reset dialogue history for new conversation"""
        self.dialogue_history = []
        logger.info("PatientAgent dialogue history reset")
    
    def respond(self, doctor_message: str) -> str:
        """
        Generate patient response to doctor's message
        
        Uses LLM with personality traits to generate response.
        
        Args:
            doctor_message: Doctor's latest message
        
        Returns:
            Patient's response message
        """
        logger.info(f"Patient generating response to doctor message")
        
        # Add doctor's message to history
        self.dialogue_history.append({
            "role": "user",  # Doctor is "user" from patient's perspective
            "content": doctor_message
        })
        
        # Build conversation messages for LLM
        messages: List[
            ChatCompletionSystemMessageParam |
            ChatCompletionUserMessageParam |
            ChatCompletionAssistantMessageParam
        ] = [
            ChatCompletionSystemMessageParam(content=self.system_prompt, role="system")
        ]
        
        # Add dialogue history
        for turn in self.dialogue_history:
            if turn["role"] == "user":
                messages.append(ChatCompletionUserMessageParam(
                    content=turn["content"],
                    role="user"
                ))
            elif turn["role"] == "assistant":
                messages.append(ChatCompletionAssistantMessageParam(
                    content=turn["content"],
                    role="assistant"
                ))
        
        # Generate patient response with retry logic
        patient_response = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Generating patient response (attempt {attempt + 1}/{self.max_retries})")
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                
                patient_response = completion.choices[0].message.content
                
                # Check if response is valid
                if patient_response is not None and len(patient_response.strip()) > 0:
                    logger.info(f"Patient generated response ({len(patient_response)} chars)")
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: API returned empty or None content")
                    last_error = "Empty or None response from API"
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                last_error = str(e)
                
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        # Handle final failure
        if patient_response is None or len(patient_response.strip()) == 0:
            error_msg = f"Failed to generate patient response after {self.max_retries} attempts. Last error: {last_error}"
            logger.error(error_msg)
            # Use a random fallback message to maintain natural conversation flow
            patient_response = random.choice(FALLBACK_MESSAGES)
            logger.info(f"Using fallback message: {patient_response[:50]}...")
        
        # Add patient's response to history
        self.dialogue_history.append({
            "role": "assistant",
            "content": patient_response
        })
        
        return patient_response
    
    def get_dialogue_history(self) -> list[dict]:
        """
        Get full dialogue history
        
        Returns:
            List of dialogue turns with role and content
        """
        return self.dialogue_history.copy()
