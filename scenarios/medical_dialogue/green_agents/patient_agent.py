"""
Patient Agent - Simulates patient with personality-driven behavior
"""

import logging
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam

logger = logging.getLogger(__name__)


class PatientAgent:
    """
    Simulates patient with personality-driven behavior
    
    Uses full system prompt (MBTI personality + background + concerns)
    that is HIDDEN from Doctor Agent
    """
    
    def __init__(self, client: OpenAI, model: str, system_prompt: str):
        """
        Initialize PatientAgent
        
        Args:
            client: OpenAI client for LLM calls
            model: Model name to use
            system_prompt: Full patient persona system prompt (includes personality)
        """
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.dialogue_history: list[dict] = []  # List of {role: str, content: str}
        
        logger.info("PatientAgent initialized with personality-driven system prompt")
    
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
        messages = [
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
        
        # Generate patient response
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        
        patient_response = completion.choices[0].message.content
        
        # Add patient's response to history
        self.dialogue_history.append({
            "role": "assistant",
            "content": patient_response
        })
        
        logger.info(f"Patient generated response ({len(patient_response)} chars)")
        return patient_response
    
    def get_dialogue_history(self) -> list[dict]:
        """
        Get full dialogue history
        
        Returns:
            List of dialogue turns with role and content
        """
        return self.dialogue_history.copy()
