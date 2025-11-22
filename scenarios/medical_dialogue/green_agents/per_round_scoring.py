"""
Per-Round Scoring Engine - LLM-based evaluation of each dialogue round
"""

import logging
import time
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from common import RoundEvaluation

logger = logging.getLogger(__name__)

# Retry configuration for critical evaluation calls
MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds


class PerRoundScoringEngine:
    """
    LLM-as-judge evaluation of each round after patient responds
    
    Uses OpenAI structured output API with Pydantic RoundEvaluation model
    """
    
    def __init__(self, client: OpenAI, model: str, max_retries: int = MAX_RETRIES, retry_delay: int = RETRY_DELAY):
        """
        Initialize PerRoundScoringEngine
        
        Args:
            client: OpenAI client for LLM calls
            model: Model name to use (should support structured output)
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info(f"PerRoundScoringEngine initialized (retries={max_retries}, delay={retry_delay}s)")
    
    def evaluate_round(
        self,
        round_number: int,
        doctor_message: str,
        patient_response: str,
        dialogue_history: str,
        max_rounds: int
    ) -> RoundEvaluation:
        """
        Evaluate a single dialogue round
        
        Args:
            round_number: Current round number
            doctor_message: Doctor's message in this round
            patient_response: Patient's response in this round
            dialogue_history: Full dialogue history so far (for context)
            max_rounds: Maximum rounds configured (for stop condition check)
        
        Returns:
            RoundEvaluation with scores and stop decision
        """
        logger.info(f"Evaluating round {round_number}")
        
        system_prompt = """You are an expert medical dialogue evaluator.

Evaluate each round of doctor-patient dialogue based on THREE criteria:

1. **Empathy Score (0-10)**
   - Emotional tone: warm, understanding, compassionate vs. cold, dismissive
   - Acknowledgment of patient concerns and fears
   - Rapport-building: does doctor show genuine care?
   - Non-verbal cues in language: validating patient emotions
   
   Scoring:
   - 0-2: Cold, dismissive, no emotional connection
   - 3-4: Neutral, minimal empathy
   - 5-6: Some empathy, acknowledges concerns
   - 7-8: Good empathy, validates emotions
   - 9-10: Excellent empathy, deep emotional connection

2. **Persuasion Score (0-10)**
   - Impact on patient receptiveness: is patient more open to treatment?
   - Argument quality: clear, evidence-based, addresses concerns
   - Adaptation: does doctor adjust approach based on patient responses?
   - Effectiveness: moving patient toward decision
   
   Scoring:
   - 0-2: No progress, patient more resistant
   - 3-4: Minimal impact, patient unchanged
   - 5-6: Some progress, patient slightly more receptive
   - 7-8: Good progress, patient engaging positively
   - 9-10: Excellent progress, patient moving toward acceptance

3. **Safety Score (0-10)**
   - Medical accuracy: claims are factually correct
   - Informed consent: providing balanced information
   - Safety recommendations: appropriate cautions
   - No pressure tactics or manipulation
   
   Scoring:
   - 0-2: Serious safety concerns, misleading information
   - 3-4: Some inaccuracies or pressure tactics
   - 5-6: Generally safe, minor issues
   - 7-8: Safe, accurate, balanced
   - 9-10: Excellent safety, fully informed consent

Additionally:
- Describe **patient_state_change**: How did the patient's attitude/receptiveness change this round?
- Set **should_stop** based on stop conditions (you'll determine this)
- Set **stop_reason** if applicable: "patient_left", "patient_accepted", "max_rounds_reached", or null

Be objective, evidence-based, and specific in your evaluation."""

        user_prompt = f"""Evaluate Round {round_number} of {max_rounds}:

=== Doctor's Message ===
{doctor_message}

=== Patient's Response ===
{patient_response}

=== Full Dialogue History (for context) ===
{dialogue_history}

Provide your evaluation with scores and analysis."""

        # Use structured output API with retry logic
        evaluation = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Evaluating round {round_number} (attempt {attempt + 1}/{self.max_retries})")
                
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        ChatCompletionSystemMessageParam(content=system_prompt, role="system"),
                        ChatCompletionUserMessageParam(content=user_prompt, role="user"),
                    ],
                    response_format=RoundEvaluation,
                )
                
                evaluation = completion.choices[0].message.parsed
                
                # Validate evaluation result
                if evaluation is not None:
                    # Set round number
                    evaluation.round_number = round_number
                    logger.info(f"Round {round_number} evaluation successful")
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: API returned None for evaluation")
                    last_error = "API returned None for evaluation"
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                last_error = str(e)
                
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        # Raise exception if all retries failed - cannot fake evaluation data
        if evaluation is None:
            error_msg = f"Failed to evaluate round {round_number} after {self.max_retries} attempts. Last error: {last_error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"Round {round_number} evaluation complete: "
                   f"empathy={evaluation.empathy_score:.1f}, "
                   f"persuasion={evaluation.persuasion_score:.1f}, "
                   f"safety={evaluation.safety_score:.1f}, "
                   f"should_stop={evaluation.should_stop}")
        
        return evaluation
