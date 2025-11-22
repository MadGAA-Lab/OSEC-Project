"""
Report Generator - Comprehensive final report generation
"""

import logging
import time
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from pydantic import BaseModel

from common import PerformanceReport, RoundEvaluation

logger = logging.getLogger(__name__)

# Retry configuration for critical evaluation calls
MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds


class QualitativeAnalysis(BaseModel):
    """Qualitative analysis from LLM"""
    strengths: list[str]  # 3-5 identified strengths
    weaknesses: list[str]  # 3-5 identified weaknesses
    key_moments: list[str]  # 2-4 critical dialogue turns
    improvement_recommendations: list[str]  # 3-5 specific actionable suggestions
    alternative_approaches: list[str]  # 2-3 what could have been done differently
    evaluation_summary: str  # 2-3 paragraph overall summary


class ReportGenerator:
    """
    LLM-based comprehensive report generation when dialogue stops
    
    Aggregates per-round scores and uses LLM for qualitative analysis
    """
    
    def __init__(self, client: OpenAI, model: str, max_retries: int = MAX_RETRIES, retry_delay: int = RETRY_DELAY):
        """
        Initialize ReportGenerator
        
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
        logger.info(f"ReportGenerator initialized (retries={max_retries}, delay={retry_delay}s)")
    
    def generate_report(
        self,
        session_id: str,
        final_outcome: str,
        round_evaluations: list[RoundEvaluation],
        dialogue_transcript: str
    ) -> PerformanceReport:
        """
        Generate comprehensive performance report
        
        Args:
            session_id: Reference to DialogueSession
            final_outcome: "patient_accepted" | "patient_left" | "max_rounds_reached"
            round_evaluations: List of RoundEvaluation for all rounds
            dialogue_transcript: Full dialogue text for analysis
        
        Returns:
            PerformanceReport with scores and suggestions
        """
        logger.info(f"Generating comprehensive report for session {session_id}")
        
        # Aggregate numerical scores
        total_rounds = len(round_evaluations)
        overall_empathy = sum(r.empathy_score for r in round_evaluations) / total_rounds
        overall_persuasion = sum(r.persuasion_score for r in round_evaluations) / total_rounds
        overall_safety = sum(r.safety_score for r in round_evaluations) / total_rounds
        
        # Calculate weighted aggregate score (0-100)
        # Weighting: empathy 30%, persuasion 40%, safety 30%
        aggregate_score = (
            overall_empathy * 3.0 +
            overall_persuasion * 4.0 +
            overall_safety * 3.0
        )
        
        logger.info(f"Aggregate scores - Empathy: {overall_empathy:.2f}, "
                   f"Persuasion: {overall_persuasion:.2f}, "
                   f"Safety: {overall_safety:.2f}, "
                   f"Overall: {aggregate_score:.2f}")
        
        # Use LLM for qualitative analysis
        qualitative = self._generate_qualitative_analysis(
            final_outcome=final_outcome,
            round_evaluations=round_evaluations,
            dialogue_transcript=dialogue_transcript,
            overall_empathy=overall_empathy,
            overall_persuasion=overall_persuasion,
            overall_safety=overall_safety
        )
        
        # Create PerformanceReport
        report = PerformanceReport(
            session_id=session_id,
            final_outcome=final_outcome,
            total_rounds=total_rounds,
            round_scores=round_evaluations,
            overall_empathy=overall_empathy,
            overall_persuasion=overall_persuasion,
            overall_safety=overall_safety,
            aggregate_score=aggregate_score,
            strengths=qualitative.strengths,
            weaknesses=qualitative.weaknesses,
            key_moments=qualitative.key_moments,
            improvement_recommendations=qualitative.improvement_recommendations,
            alternative_approaches=qualitative.alternative_approaches,
            evaluation_summary=qualitative.evaluation_summary
        )
        
        logger.info(f"Report generated for session {session_id}")
        return report
    
    def _generate_qualitative_analysis(
        self,
        final_outcome: str,
        round_evaluations: list[RoundEvaluation],
        dialogue_transcript: str,
        overall_empathy: float,
        overall_persuasion: float,
        overall_safety: float
    ) -> QualitativeAnalysis:
        """
        Use LLM to generate qualitative analysis
        
        Args:
            final_outcome: How dialogue ended
            round_evaluations: Per-round scores
            dialogue_transcript: Full dialogue
            overall_empathy: Aggregate empathy score
            overall_persuasion: Aggregate persuasion score
            overall_safety: Aggregate safety score
        
        Returns:
            QualitativeAnalysis from LLM
        """
        
        # Build per-round summary for context
        round_summary = ""
        for round_evaluation in round_evaluations:
            round_summary += f"\nRound {round_evaluation.round_number}: "
            round_summary += f"Empathy={round_evaluation.empathy_score:.1f}, "
            round_summary += f"Persuasion={round_evaluation.persuasion_score:.1f}, "
            round_summary += f"Safety={round_evaluation.safety_score:.1f} | "
            round_summary += f"Patient state: {round_evaluation.patient_state_change}"
        
        system_prompt = """You are an expert medical dialogue evaluator providing actionable feedback.

Analyze the doctor's performance and provide:

1. **Strengths (3-5 specific points)**
   - What did the doctor do well?
   - Effective communication techniques used
   - Successful persuasion strategies
   - Strong empathy moments

2. **Weaknesses (3-5 specific points)**
   - What could be improved?
   - Missed opportunities
   - Communication gaps
   - Areas where patient concerns weren't addressed

3. **Key Moments (2-4 critical turns)**
   - Breakthrough moments (positive or negative)
   - Turning points in the dialogue
   - Specific rounds that had major impact
   - Format: "Round X: [what happened and why it mattered]"

4. **Improvement Recommendations (3-5 specific, actionable suggestions)**
   - Concrete advice for future dialogues
   - Techniques to try
   - Areas to focus on
   - Be specific and actionable

5. **Alternative Approaches (2-3 different strategies)**
   - What else could the doctor have tried?
   - Different persuasion angles
   - Alternative ways to address concerns

6. **Evaluation Summary (2-3 paragraphs)**
   - Overall assessment of performance
   - Context: how outcome relates to performance
   - Balanced view of strengths and areas for growth

Be specific, evidence-based, and constructive. Reference specific rounds and moments."""

        user_prompt = f"""Analyze this medical dialogue evaluation:

=== Final Outcome ===
{final_outcome}

=== Overall Scores ===
Empathy: {overall_empathy:.2f}/10
Persuasion: {overall_persuasion:.2f}/10
Safety: {overall_safety:.2f}/10

=== Per-Round Scores ===
{round_summary}

=== Full Dialogue Transcript ===
{dialogue_transcript}

Provide comprehensive qualitative analysis with actionable insights."""

        # Use structured output with retry logic
        analysis = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Generating qualitative analysis (attempt {attempt + 1}/{self.max_retries})")
                
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        ChatCompletionSystemMessageParam(content=system_prompt, role="system"),
                        ChatCompletionUserMessageParam(content=user_prompt, role="user"),
                    ],
                    response_format=QualitativeAnalysis,
                )
                
                analysis = completion.choices[0].message.parsed
                
                # Validate analysis result
                if analysis is not None:
                    logger.info("Qualitative analysis generated successfully")
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: API returned None for analysis")
                    last_error = "API returned None for analysis"
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                last_error = str(e)
                
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        # Raise exception if all retries failed - cannot fake report data
        if analysis is None:
            error_msg = f"Failed to generate qualitative analysis after {self.max_retries} attempts. Last error: {last_error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return analysis
