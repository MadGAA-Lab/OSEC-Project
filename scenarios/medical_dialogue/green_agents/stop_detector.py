"""
Stop Condition Detector - Determines if dialogue should terminate
"""

import logging
import time
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Retry configuration for critical evaluation calls
MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds


class StopDecision(BaseModel):
    """Stop decision output from LLM"""
    should_stop: bool
    stop_reason: str | None  # "patient_left" | "patient_accepted" | "max_rounds_reached" | null
    confidence: str  # "high" | "medium" | "low"
    reasoning: str  # Explanation for the decision


class StopConditionDetector:
    """
    LLM-based classification to determine if dialogue should terminate
    
    Checks for:
    1. Patient left (explicit refusal, strong negative sentiment, disengagement)
    2. Patient accepted (explicit acceptance, strong positive commitment)
    3. Max rounds reached (current round >= max_rounds)
    """
    
    def __init__(self, client: OpenAI, model: str, max_retries: int = MAX_RETRIES, retry_delay: int = RETRY_DELAY):
        """
        Initialize StopConditionDetector
        
        Args:
            client: OpenAI client for LLM calls
            model: Model name to use
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"StopConditionDetector initialized (retries={max_retries}, delay={retry_delay}s)")
    
    def should_stop(
        self,
        round_number: int,
        patient_response: str,
        dialogue_history: str,
        max_rounds: int
    ) -> tuple[bool, str | None]:
        """
        Determine if dialogue should stop
        
        Args:
            round_number: Current round number
            patient_response: Patient's latest response
            dialogue_history: Full dialogue history
            max_rounds: Maximum rounds configured
        
        Returns:
            tuple: (should_stop: bool, stop_reason: str | None)
        """
        logger.info(f"Checking stop conditions for round {round_number}/{max_rounds}")
        
        # Check max rounds first (rule-based)
        if round_number >= max_rounds:
            logger.info(f"Max rounds reached: {round_number}/{max_rounds}")
            return True, "max_rounds_reached"
        
        # Use LLM to detect patient left or accepted
        system_prompt = """You are analyzing a doctor-patient dialogue to detect if it should terminate.

Analyze the patient's latest response and determine if:

1. **Patient Left (should stop)**
   - Explicit refusal: "I'm not doing this surgery", "I'm leaving", "I want a second opinion"
   - Strong negative sentiment: anger, frustration, distrust
   - Disengagement: "I don't want to talk about this anymore"
   - Clear rejection of treatment despite doctor's efforts
   
2. **Patient Accepted (should stop)**
   - Explicit acceptance: "I'll do the surgery", "Let's schedule it", "I agree to the treatment"
   - Strong positive commitment: "You've convinced me", "I'm ready"
   - Clear agreement to proceed with recommended treatment
   
3. **Should Continue (don't stop)**
   - Patient is still engaged and considering
   - Asking questions, expressing concerns, but not final decision
   - Uncertain but willing to continue dialogue

Be conservative: only stop if there's clear evidence of patient leaving OR accepting.
If patient is still engaged and considering, mark should_stop as false."""

        user_prompt = f"""Analyze this dialogue to determine if it should stop:

Round: {round_number} of {max_rounds}

=== Patient's Latest Response ===
{patient_response}

=== Full Dialogue History ===
{dialogue_history}

Determine:
- should_stop: true or false
- stop_reason: "patient_left", "patient_accepted", or null
- confidence: "high", "medium", or "low"
- reasoning: explain your decision"""

        # Use structured output with retry logic
        decision = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Checking stop condition (attempt {attempt + 1}/{self.max_retries})")
                
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        ChatCompletionSystemMessageParam(content=system_prompt, role="system"),
                        ChatCompletionUserMessageParam(content=user_prompt, role="user"),
                    ],
                    response_format=StopDecision,
                )
                
                decision = completion.choices[0].message.parsed
                
                # Validate decision result
                if decision is not None:
                    logger.info(f"Stop decision successful: should_stop={decision.should_stop}")
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: API returned None for decision")
                    last_error = "API returned None for decision"
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                last_error = str(e)
                
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        # Raise exception if all retries failed - cannot fake stop decision
        if decision is None:
            error_msg = f"Failed to determine stop condition after {self.max_retries} attempts. Last error: {last_error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"Stop decision: should_stop={decision.should_stop}, "
                   f"reason={decision.stop_reason}, confidence={decision.confidence}")
        logger.debug(f"Stop reasoning: {decision.reasoning}")
        
        return decision.should_stop, decision.stop_reason
