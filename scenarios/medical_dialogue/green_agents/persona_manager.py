"""
Persona Manager - Utility to load prompt templates for patient personas
"""

import os
from pathlib import Path


# All 16 MBTI types
MBTI_TYPES = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
]

GENDERS = ["male", "female"]
MEDICAL_CASES = ["pneumothorax", "lung_cancer"]


class PersonaManager:
    """Manages prompt template files for patient personas"""
    
    def __init__(self, prompts_dir: str | None = None):
        """
        Initialize PersonaManager
        
        Args:
            prompts_dir: Path to prompts directory. If None, uses default location
        """
        if prompts_dir is None:
            # Default: scenarios/medical_dialogue/prompts/
            current_file = Path(__file__)
            scenario_dir = current_file.parent.parent
            self.prompts_dir = scenario_dir / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
        
        self.mbti_dir = self.prompts_dir / "mbti"
        self.gender_dir = self.prompts_dir / "gender"
        self.cases_dir = self.prompts_dir / "cases"
    
    def parse_persona_id(self, persona_id: str) -> tuple[str, str, str]:
        """
        Parse persona_id into components
        
        Args:
            persona_id: e.g., "INTJ_M_PNEUMO" or "ESFP_F_LUNG"
        
        Returns:
            tuple: (mbti_type, gender_code, case_code)
        """
        parts = persona_id.split("_")
        if len(parts) != 3:
            raise ValueError(f"Invalid persona_id format: {persona_id}. Expected format: MBTI_GENDER_CASE")
        
        mbti, gender_code, case_code = parts
        
        # Validate components
        if mbti.upper() not in MBTI_TYPES:
            raise ValueError(f"Invalid MBTI type: {mbti}")
        
        if gender_code.upper() not in ["M", "F"]:
            raise ValueError(f"Invalid gender code: {gender_code}. Use M or F")
        
        if case_code.upper() not in ["PNEUMO", "LUNG"]:
            raise ValueError(f"Invalid case code: {case_code}. Use PNEUMO or LUNG")
        
        return mbti.upper(), gender_code.upper(), case_code.upper()
    
    def get_prompt_paths(self, persona_id: str) -> dict[str, Path]:
        """
        Get file paths for all prompt templates for a persona
        
        Args:
            persona_id: e.g., "INTJ_M_PNEUMO"
        
        Returns:
            dict with keys: 'mbti', 'gender', 'case'
        """
        mbti, gender_code, case_code = self.parse_persona_id(persona_id)
        
        # Map codes to filenames
        gender = "male" if gender_code == "M" else "female"
        case = "pneumothorax" if case_code == "PNEUMO" else "lung_cancer"
        
        return {
            "mbti": self.mbti_dir / f"{mbti.lower()}.txt",
            "gender": self.gender_dir / f"{gender}.txt",
            "case": self.cases_dir / f"{case}.txt"
        }
    
    def load_prompt_templates(self, persona_id: str) -> dict[str, str]:
        """
        Load all prompt templates for a persona
        
        Args:
            persona_id: e.g., "INTJ_M_PNEUMO"
        
        Returns:
            dict with keys: 'mbti', 'gender', 'case' containing prompt text
        """
        paths = self.get_prompt_paths(persona_id)
        prompts = {}
        
        for key, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Prompt file not found: {path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                prompts[key] = f.read().strip()
        
        return prompts
    
    def get_all_persona_ids(self) -> list[str]:
        """
        Generate all possible persona IDs (64 combinations)
        
        Returns:
            list of all persona_ids
        """
        persona_ids = []
        for mbti in MBTI_TYPES:
            for gender_code in ["M", "F"]:
                for case_code in ["PNEUMO", "LUNG"]:
                    persona_ids.append(f"{mbti}_{gender_code}_{case_code}")
        return persona_ids
    
    def expand_persona_ids(self, persona_ids: list[str]) -> list[str]:
        """
        Expand persona_ids list, handling "all" keyword
        
        Args:
            persona_ids: List that may contain "all" or specific persona IDs
        
        Returns:
            Expanded list of specific persona IDs
        """
        if "all" in persona_ids:
            return self.get_all_persona_ids()
        return persona_ids
