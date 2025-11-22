from pydantic import BaseModel
from typing import Literal
from datetime import datetime

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)


# ==================== Data Models ====================

class PatientPersona(BaseModel):
    """Minimal patient persona config - details generated dynamically"""
    persona_id: str  # e.g., "INTJ_M_PNEUMO"
    mbti_type: str  # MBTI personality type (16 options: INTJ, ESFP, etc.)
    gender: str  # "male" or "female"
    medical_case: str  # "pneumothorax" or "lung_cancer"
    system_prompt: str  # Generated complete system prompt for patient agent


class PatientClinicalInfo(BaseModel):
    """Partial patient information provided to Doctor Agent (NO personality traits)"""
    age: int  # Patient age
    gender: str  # "male" or "female"
    medical_case: str  # "pneumothorax" or "lung_cancer"
    symptoms: str  # Brief symptom description
    diagnosis: str  # Medical diagnosis
    recommended_treatment: str  # Recommended surgical procedure
    case_background: str  # Clinical facts about the case


class MedicalCase(BaseModel):
    """Medical condition template - used to generate case-specific prompts"""
    case_id: str  # "pneumothorax" or "lung_cancer"
    case_prompt: str  # Complete text description


class RoundEvaluation(BaseModel):
    """Per-round evaluation results"""
    round_number: int  # Which round was evaluated
    empathy_score: float  # Emotional support quality (0-10)
    persuasion_score: float  # Persuasion effectiveness (0-10)
    safety_score: float  # Medical safety and accuracy (0-10)
    patient_state_change: str  # Description of how patient's attitude changed
    should_stop: bool  # Whether dialogue should terminate
    stop_reason: str | None  # "patient_left" | "patient_accepted" | "max_rounds_reached" | null


class DialogueTurn(BaseModel):
    """Single turn in doctor-patient dialogue with per-round evaluation"""
    turn_number: int  # Sequential turn number in dialogue
    speaker: Literal["doctor", "patient"]  # Speaker identifier
    message: str  # Dialogue message content
    timestamp: str  # ISO 8601 timestamp
    round_evaluation: RoundEvaluation | None = None  # Evaluation results if round complete


class DialogueSession(BaseModel):
    """Complete dialogue session record with per-round evaluations"""
    session_id: str  # Unique session identifier
    persona_id: str  # Patient persona identifier
    doctor_agent_url: str  # Purple agent endpoint
    start_time: str  # ISO 8601 timestamp
    end_time: str | None = None  # ISO 8601 timestamp
    turns: list[DialogueTurn] = []  # All dialogue turns
    total_rounds: int = 0  # Number of complete rounds
    final_outcome: str | None = None  # "patient_accepted" | "patient_left" | "max_rounds_reached"
    stop_reason: str | None = None  # Why dialogue terminated


class PerformanceReport(BaseModel):
    """Comprehensive final report with per-round and overall scores"""
    session_id: str  # Reference to DialogueSession
    final_outcome: str  # "patient_accepted" | "patient_left" | "max_rounds_reached"
    total_rounds: int  # Number of rounds completed
    
    # Per-Round Breakdown
    round_scores: list[RoundEvaluation]  # Score for each round
    
    # Overall Aggregate Scores
    overall_empathy: float  # Mean empathy across all rounds (0-10)
    overall_persuasion: float  # Mean persuasion across all rounds (0-10)
    overall_safety: float  # Mean safety across all rounds (0-10)
    aggregate_score: float  # Weighted overall score (0-100)
    
    # Qualitative Analysis
    strengths: list[str]  # Identified strengths
    weaknesses: list[str]  # Identified weaknesses
    key_moments: list[str]  # Critical dialogue turns
    
    # Actionable Suggestions
    improvement_recommendations: list[str]  # Specific advice
    alternative_approaches: list[str]  # What could have been done differently
    
    # Summary
    evaluation_summary: str  # Overall text summary


class MedicalEvalResult(BaseModel):
    """Complete evaluation results across multiple personas"""
    assessment_id: str  # Unique assessment identifier
    doctor_agent_url: str  # Evaluated purple agent
    timestamp: str  # ISO 8601 timestamp
    sessions: list[DialogueSession]  # All dialogue sessions conducted
    reports: list[PerformanceReport]  # Comprehensive reports per session
    mean_aggregate_score: float  # Average score across all sessions
    overall_summary: str  # Text summary across all personas


# ==================== Agent Card Helpers ====================

def medical_judge_agent_card(agent_name: str, card_url: str) -> AgentCard:
    """Generate agent card for Medical Dialogue Judge"""
    skill = AgentSkill(
        id='evaluate_medical_dialogue',
        name='Evaluate medical dialogue agents',
        description='Orchestrate and evaluate doctor-patient dialogue across multiple patient personas with MBTI personality types.',
        tags=['medical', 'evaluation', 'dialogue'],
        examples=["""
{
  "participants": {
    "doctor": "http://127.0.0.1:9019"
  },
  "config": {
    "persona_ids": ["INTJ_M_PNEUMO", "ESFP_F_LUNG"],
    "max_rounds": 5
  }
}
""", """
{
  "participants": {
    "doctor": "http://127.0.0.1:9019"
  },
  "config": {
    "persona_ids": ["all"],
    "max_rounds": 5
  }
}
"""]
    )
    agent_card = AgentCard(
        name=agent_name,
        description='Evaluates doctor agents ability to persuade patients to accept surgical treatment across diverse patient personas (16 MBTI types × 2 genders × 2 medical conditions).',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card
