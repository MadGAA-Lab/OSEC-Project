"""
Patient Constructor - Generates patient persona system prompts from templates
"""

import json
import logging
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from persona_manager import PersonaManager
from common import PatientPersona, PatientClinicalInfo

logger = logging.getLogger(__name__)


class PatientConstructor:
    """Constructs patient system prompts from MBTI/gender/case templates"""
    
    def __init__(self, client: OpenAI, model: str, persona_manager: PersonaManager | None = None):
        """
        Initialize PatientConstructor
        
        Args:
            client: OpenAI client for LLM calls
            model: Model name to use
            persona_manager: PersonaManager instance (creates new if None)
        """
        self.client = client
        self.model = model
        self.persona_manager = persona_manager or PersonaManager()
    
    def construct_patient_persona(self, persona_id: str) -> PatientPersona:
        """
        Generate complete patient persona with system prompt
        
        Args:
            persona_id: e.g., "INTJ_M_PNEUMO"
        
        Returns:
            PatientPersona with generated system_prompt
        """
        logger.info(f"Constructing patient persona: {persona_id}")
        
        # Parse persona components
        mbti, gender_code, case_code = self.persona_manager.parse_persona_id(persona_id)
        gender = "male" if gender_code == "M" else "female"
        medical_case = "pneumothorax" if case_code == "PNEUMO" else "lung_cancer"
        
        # Load prompt templates
        templates = self.persona_manager.load_prompt_templates(persona_id)
        
        # Generate system prompt using LLM
        system_prompt = self._generate_system_prompt(
            mbti_prompt=templates["mbti"],
            gender_prompt=templates["gender"],
            case_prompt=templates["case"],
            mbti_type=mbti,
            gender=gender,
            medical_case=medical_case
        )
        
        # Create PatientPersona
        persona = PatientPersona(
            persona_id=persona_id,
            mbti_type=mbti,
            gender=gender,
            medical_case=medical_case,
            system_prompt=system_prompt
        )
        
        logger.info(f"Successfully constructed persona: {persona_id}")
        return persona
    
    def _generate_system_prompt(
        self,
        mbti_prompt: str,
        gender_prompt: str,
        case_prompt: str,
        mbti_type: str,
        gender: str,
        medical_case: str
    ) -> str:
        """
        Use LLM to generate coherent patient system prompt from templates
        
        Args:
            mbti_prompt: MBTI personality description
            gender_prompt: Gender context
            case_prompt: Medical case details
            mbti_type: MBTI type code
            gender: male/female
            medical_case: pneumothorax/lung_cancer
        
        Returns:
            Generated system prompt string
        """
        
        system_msg = """You are a helpful assistant creating a patient character description for a medical dialogue simulation.

Your task: synthesize the following elements into a coherent, realistic patient persona description:
1. MBTI personality traits and communication patterns
2. Gender-specific background considerations  
3. Medical case details (diagnosis, symptoms, treatment)

Create a persona description that includes:
- Age (realistic for the condition, 35-65 range)
- Occupation (aligned with personality type)
- Background story (family, lifestyle, values)
- Personality-driven concerns and fears about the medical situation
- Communication style and behavioral patterns
- How this person responds to different persuasion approaches

IMPORTANT: Write the description in SECOND PERSON ("You are...") as if directly instructing someone to roleplay this patient.
The description will be used as a system prompt for an AI agent to roleplay this patient in a medical consultation.

Make the character natural, coherent, and realistic - a believable person with depth and complexity.
Output 300-500 words."""

        user_msg = f"""Create a patient character description by combining these elements:

=== MBTI Personality Type: {mbti_type} ===
{mbti_prompt}

=== Gender: {gender} ===
{gender_prompt}

=== Medical Case: {medical_case} ===
{case_prompt}

Write a cohesive patient persona in second person ("You are...") that naturally integrates all three elements.
This will be used as a system prompt to roleplay this patient in a medical dialogue."""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                ChatCompletionSystemMessageParam(content=system_msg, role="system"),
                ChatCompletionUserMessageParam(content=user_msg, role="user"),
            ],
        )
        
        system_prompt = completion.choices[0].message.content
        
        # Add explicit roleplay instructions to ensure the agent stays in character
        roleplay_footer = """

---
ROLEPLAY INSTRUCTIONS:
You are roleplaying this patient character in a medical consultation with a doctor. Stay fully in character throughout the entire conversation. Respond naturally as this patient would, expressing their concerns, asking questions, and reacting to the doctor's explanations based on your personality, background, and medical situation. Do not break character or discuss the roleplay itself."""
        
        system_prompt = system_prompt + roleplay_footer
        return system_prompt
    
    def extract_clinical_info(self, persona: PatientPersona) -> PatientClinicalInfo:
        """
        Extract PatientClinicalInfo from persona (subset for Doctor Agent)
        
        This extracts only clinical facts, NOT personality traits or concerns.
        Uses LLM to parse the system prompt and extract structured clinical info.
        
        Args:
            persona: PatientPersona with full system prompt
        
        Returns:
            PatientClinicalInfo (clinical facts only)
        """
        logger.info(f"Extracting clinical info from persona: {persona.persona_id}")
        
        system_msg = """You are extracting clinical information from a patient persona description.
Extract ONLY factual medical information:
- Age (number)
- Gender (male/female)
- Medical case (pneumothorax/lung_cancer)
- Symptoms (brief description)
- Diagnosis (medical diagnosis)
- Recommended treatment (surgical procedure)
- Case background (clinical facts, risks, benefits)

DO NOT include personality traits, concerns, fears, or behavioral patterns.
Return a JSON object with these exact fields: age, gender, medical_case, symptoms, diagnosis, recommended_treatment, case_background"""

        user_msg = f"""Extract clinical information from this patient persona:

{persona.system_prompt}

Return JSON only."""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                ChatCompletionSystemMessageParam(content=system_msg, role="system"),
                ChatCompletionUserMessageParam(content=user_msg, role="user"),
            ],
            response_format={"type": "json_object"},
        )
        
        # Parse JSON response
        response_text = completion.choices[0].message.content
        clinical_data = json.loads(response_text)
        
        # Create PatientClinicalInfo
        clinical_info = PatientClinicalInfo(
            age=int(clinical_data["age"]),
            gender=clinical_data["gender"],
            medical_case=clinical_data["medical_case"],
            symptoms=clinical_data["symptoms"],
            diagnosis=clinical_data["diagnosis"],
            recommended_treatment=clinical_data["recommended_treatment"],
            case_background=clinical_data["case_background"]
        )
        
        logger.info(f"Extracted clinical info for persona: {persona.persona_id}")
        return clinical_info
