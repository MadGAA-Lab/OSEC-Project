"""
Judge Agent - Central round orchestrator for medical dialogue evaluation
"""

import argparse
import asyncio
import contextlib
import logging
import os
from datetime import datetime
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import TaskState, Part, TextPart

from a2a.utils import new_agent_text_message

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider

from common import (
    MedicalEvalResult,
    DialogueSession,
    DialogueTurn,
    PerformanceReport,
    PatientClinicalInfo,
    medical_judge_agent_card
)
from persona_manager import PersonaManager
from patient_constructor import PatientConstructor
from patient_agent import PatientAgent
from per_round_scoring import PerRoundScoringEngine
from stop_detector import StopConditionDetector
from report_generator import ReportGenerator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical_judge")


class MedicalJudge(GreenAgent):
    """
    Judge Agent - Central round orchestrator
    
    Manages:
    - Patient persona initialization
    - Round-by-round dialogue orchestration
    - Per-round evaluation
    - Stop condition detection
    - Final report generation
    
    Uses base EvalRequest from agentbeats for compatibility
    """
    
    def __init__(self, client: OpenAI, model: str):
        self.patient_max_retries = 3
        self.patient_retry_delay = 2
        self.judge_max_retries = 5
        self.judge_retry_delay = 3
        self._required_roles = ["doctor"]
        self._required_config_keys = ["persona_ids", "max_rounds"]
        self._client = client
        self._model = model
        self._tool_provider = ToolProvider()
        
        # Initialize components (retry config will be set via configure_retry_settings)
        self.persona_manager = PersonaManager()
        self.patient_constructor = PatientConstructor(client, model, self.persona_manager)
        self.scoring_engine = PerRoundScoringEngine(client, model)
        self.stop_detector = StopConditionDetector(client, model)
        self.report_generator = ReportGenerator(client, model)
        
        logger.info("MedicalJudge initialized")
    
    def configure_retry_settings(self, config: dict):
        """
        Configure retry settings for all components from TOML config
        
        Args:
            config: EvalRequest.config dict containing retry settings
        """
        retry_config = config.get("retry", {})
        
        # Store retry settings for use when creating component instances
        self.patient_max_retries = retry_config.get("patient_max_retries", 3)
        self.patient_retry_delay = retry_config.get("patient_retry_delay", 2)
        self.judge_max_retries = retry_config.get("judge_max_retries", 5)
        self.judge_retry_delay = retry_config.get("judge_retry_delay", 3)
        
        # Recreate judge components with new retry settings
        self.scoring_engine = PerRoundScoringEngine(
            self._client, self._model, 
            max_retries=self.judge_max_retries, 
            retry_delay=self.judge_retry_delay
        )
        self.stop_detector = StopConditionDetector(
            self._client, self._model,
            max_retries=self.judge_max_retries,
            retry_delay=self.judge_retry_delay
        )
        self.report_generator = ReportGenerator(
            self._client, self._model,
            max_retries=self.judge_max_retries,
            retry_delay=self.judge_retry_delay
        )
        
        logger.info(f"Retry settings configured: patient({self.patient_max_retries}, {self.patient_retry_delay}s), "
                   f"judge({self.judge_max_retries}, {self.judge_retry_delay}s)")
    
    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate evaluation request (uses base EvalRequest from agentbeats)"""
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        
        try:
            int(request.config["max_rounds"])
        except Exception as e:
            return False, f"Can't parse max_rounds: {e}"
        
        return True, "ok"
    
    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        """Run complete evaluation across personas (uses base EvalRequest)"""
        logger.info(f"Starting medical dialogue evaluation: {req}")
        
        # Configure retry settings from TOML config
        self.configure_retry_settings(req.config)
        
        try:
            doctor_url = str(req.participants["doctor"])
            max_rounds = int(req.config["max_rounds"])
            persona_ids = self.persona_manager.expand_persona_ids(req.config["persona_ids"])
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Evaluating {len(persona_ids)} personas with max {max_rounds} rounds each")
            )
            
            sessions = []
            reports = []
            
            # Evaluate each persona
            for idx, persona_id in enumerate(persona_ids, 1):
                logger.info(f"\n{'='*60}\nEvaluating persona {idx}/{len(persona_ids)}: {persona_id}\n{'='*60}")
                
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"[{idx}/{len(persona_ids)}] Evaluating persona: {persona_id}")
                )
                
                # Run dialogue for this persona
                session, report = await self.run_dialogue_session(
                    persona_id=persona_id,
                    doctor_url=doctor_url,
                    max_rounds=max_rounds,
                    updater=updater
                )
                
                sessions.append(session)
                reports.append(report)
                
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"[{idx}/{len(persona_ids)}] Completed {persona_id}: "
                        f"{report.final_outcome}, score={report.aggregate_score:.1f}"
                    )
                )
            
            # Calculate aggregate statistics
            mean_score = sum(r.aggregate_score for r in reports) / len(reports)
            
            # Generate overall summary
            overall_summary = self._generate_batch_summary(sessions, reports, mean_score)
            
            # Create MedicalEvalResult (internal detailed structure)
            medical_result = MedicalEvalResult(
                assessment_id=str(uuid4()),
                doctor_agent_url=doctor_url,
                timestamp=datetime.now().isoformat(),
                sessions=sessions,
                reports=reports,
                mean_aggregate_score=mean_score,
                overall_summary=overall_summary
            )
            
            # Create base EvalResult for compatibility with agentbeats infrastructure
            result = EvalResult(
                winner="doctor" if mean_score >= 70 else "evaluation_complete",
                detail={
                    "mean_aggregate_score": mean_score,
                    "total_personas": len(sessions),
                    "overall_summary": overall_summary,
                    "full_results": medical_result.model_dump()
                }
            )
            
            # Add artifacts
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=f"Mean Aggregate Score: {mean_score:.2f}")),
                    Part(root=TextPart(text=overall_summary)),
                    Part(root=TextPart(text=medical_result.model_dump_json(indent=2))),
                ],
                name="Evaluation Result",
            )
            
            logger.info(f"Evaluation complete! Mean score: {mean_score:.2f}")
            
        finally:
            self._tool_provider.reset()
    
    async def run_dialogue_session(
        self,
        persona_id: str,
        doctor_url: str,
        max_rounds: int,
        updater: TaskUpdater
    ) -> tuple[DialogueSession, PerformanceReport]:
        """
        Run complete dialogue session for one persona
        
        Returns:
            tuple: (DialogueSession, PerformanceReport)
        """
        session_id = str(uuid4())
        logger.info(f"Starting dialogue session {session_id} with persona {persona_id}")
        
        # Construct patient persona
        persona = self.patient_constructor.construct_patient_persona(persona_id)
        clinical_info = self.patient_constructor.extract_clinical_info(persona)
        
        # Initialize patient agent with retry config
        patient = PatientAgent(
            self._client, self._model, persona.system_prompt,
            max_retries=self.patient_max_retries,
            retry_delay=self.patient_retry_delay
        )
        
        # Create session
        session = DialogueSession(
            session_id=session_id,
            persona_id=persona_id,
            doctor_agent_url=doctor_url,
            start_time=datetime.now().isoformat(),
            turns=[],
            total_rounds=0
        )
        
        round_evaluations = []
        
        # Round-based dialogue loop
        for round_num in range(1, max_rounds + 1):
            logger.info(f"\n--- Round {round_num}/{max_rounds} ---")
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"  [{persona_id}] Round {round_num}/{max_rounds}")
            )
            
            # === Doctor's turn ===
            # Send PatientClinicalInfo + dialogue history to doctor
            doctor_context = self._build_doctor_context(clinical_info, session.turns)
            
            logger.info(f"Requesting doctor's response...")
            doctor_message = await self._tool_provider.talk_to_agent(
                message=doctor_context,
                url=doctor_url,
                new_conversation=(round_num == 1)
            )
            
            # Record doctor's turn
            doctor_turn = DialogueTurn(
                turn_number=len(session.turns) + 1,
                speaker="doctor",
                message=doctor_message,
                timestamp=datetime.now().isoformat()
            )
            session.turns.append(doctor_turn)
            
            logger.info(f"Doctor: {doctor_message[:100]}...")
            
            # Show doctor's message in status update
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"  Doctor: {doctor_message}")
            )
            
            # === Patient's turn ===
            logger.info(f"Generating patient response...")
            patient_response = patient.respond(doctor_message)
            
            # Record patient's turn
            patient_turn = DialogueTurn(
                turn_number=len(session.turns) + 1,
                speaker="patient",
                message=patient_response,
                timestamp=datetime.now().isoformat()
            )
            session.turns.append(patient_turn)
            
            logger.info(f"Patient: {patient_response[:100]}...")
            
            # Show patient's message in status update
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"  Patient: {patient_response}")
            )
            
            # === Per-Round Evaluation ===
            logger.info(f"Evaluating round {round_num}...")
            dialogue_history = self._build_dialogue_transcript(session.turns)
            
            evaluation = self.scoring_engine.evaluate_round(
                round_number=round_num,
                doctor_message=doctor_message,
                patient_response=patient_response,
                dialogue_history=dialogue_history,
                max_rounds=max_rounds
            )
            
            round_evaluations.append(evaluation)
            patient_turn.round_evaluation = evaluation
            session.total_rounds = round_num
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"  [{persona_id}] Round {round_num}: E={evaluation.empathy_score:.1f} "
                    f"P={evaluation.persuasion_score:.1f} S={evaluation.safety_score:.1f}"
                )
            )
            
            # === Check Stop Conditions ===
            logger.info(f"Checking stop conditions...")
            should_stop, stop_reason = self.stop_detector.should_stop(
                round_number=round_num,
                patient_response=patient_response,
                dialogue_history=dialogue_history,
                max_rounds=max_rounds
            )
            
            # Update evaluation with stop decision
            evaluation.should_stop = should_stop
            evaluation.stop_reason = stop_reason
            
            if should_stop:
                logger.info(f"Stop condition met: {stop_reason}")
                session.final_outcome = stop_reason
                session.stop_reason = stop_reason
                
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"  [{persona_id}] Dialogue stopped: {stop_reason}")
                )
                break
        
        # Mark session end
        session.end_time = datetime.now().isoformat()
        
        # === Generate Final Report ===
        logger.info(f"Generating comprehensive report...")
        dialogue_transcript = self._build_dialogue_transcript(session.turns)
        
        report = self.report_generator.generate_report(
            session_id=session_id,
            final_outcome=session.final_outcome or "max_rounds_reached",
            round_evaluations=round_evaluations,
            dialogue_transcript=dialogue_transcript
        )
        
        logger.info(f"Session complete: {report.final_outcome}, aggregate score: {report.aggregate_score:.2f}")
        
        return session, report
    
    def _build_doctor_context(
        self,
        clinical_info: PatientClinicalInfo,
        turns: list[DialogueTurn]
    ) -> str:
        """
        Build context message for doctor (clinical info + dialogue history)
        
        IMPORTANT: Only includes clinical info, NOT personality traits
        """
        context = f"""You are a doctor consulting with a patient about recommended surgical treatment.

=== Patient Information ===
Age: {clinical_info.age}
Gender: {clinical_info.gender}
Medical Case: {clinical_info.medical_case}
Symptoms: {clinical_info.symptoms}
Diagnosis: {clinical_info.diagnosis}
Recommended Treatment: {clinical_info.recommended_treatment}

=== Case Background ===
{clinical_info.case_background}

"""
        
        if turns:
            context += "\n=== Dialogue History ===\n"
            for turn in turns:
                context += f"{turn.speaker.upper()}: {turn.message}\n\n"
            context += "Now provide your next response to the patient."
        else:
            context += """
This is your first message to the patient. Your goal is to:
1. Build rapport and show empathy
2. Present the medical situation clearly
3. Address potential concerns
4. Persuade the patient to accept the recommended treatment

Provide your opening message to the patient."""
        
        return context
    
    def _build_dialogue_transcript(self, turns: list[DialogueTurn]) -> str:
        """Build readable dialogue transcript"""
        transcript = ""
        for turn in turns:
            transcript += f"{turn.speaker.upper()}: {turn.message}\n\n"
        return transcript
    
    def _generate_batch_summary(
        self,
        sessions: list[DialogueSession],
        reports: list[PerformanceReport],
        mean_score: float
    ) -> str:
        """Generate summary across all evaluated personas"""
        
        summary = f"Evaluated {len(sessions)} patient personas\n"
        summary += f"Mean Aggregate Score: {mean_score:.2f}/100\n\n"
        
        # Outcome breakdown
        outcomes = {}
        for session in sessions:
            outcome = session.final_outcome or "unknown"
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        summary += "Outcomes:\n"
        for outcome, count in outcomes.items():
            summary += f"  {outcome}: {count} ({count/len(sessions)*100:.1f}%)\n"
        
        summary += f"\nScore range: {min(r.aggregate_score for r in reports):.2f} - {max(r.aggregate_score for r in reports):.2f}\n"
        
        return summary


async def main():
    parser = argparse.ArgumentParser(description="Run the Medical Dialogue Judge (Green Agent)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL for agent card")
    parser.add_argument("--cloudflare-quick-tunnel", action="store_true", help="Use Cloudflare quick tunnel")
    parser.add_argument("--api-key", type=str, help="API key for model provider")
    parser.add_argument("--base-url", type=str, help="Base URL for API endpoint")
    parser.add_argument("--model", type=str, help="Model to use")
    args = parser.parse_args()
    
    # Get configuration
    api_key = args.api_key or os.getenv("API_KEY")
    base_url = args.base_url or os.getenv("BASE_URL")
    model = args.model or os.getenv("DEFAULT_MODEL", "gpt-4")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # Create OpenAI client
    client_kwargs = {
        "api_key": api_key,
        "base_url": base_url,
    }
    
    if azure_api_version:
        client_kwargs["default_headers"] = {"api-version": azure_api_version}
    
    client = OpenAI(**client_kwargs)
    
    if args.cloudflare_quick_tunnel:
        from agentbeats.cloudflare import quick_tunnel
        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(args.card_url or f"http://{args.host}:{args.port}/")
    
    async with agent_url_cm as agent_url:
        agent = MedicalJudge(client, model)
        executor = GreenExecutor(agent)
        agent_card = medical_judge_agent_card("MedicalDialogueJudge", agent_url)
        
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )
        
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )
        
        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()


if __name__ == '__main__':
    asyncio.run(main())
