# Medical Dialogue Evaluation Scenario

This scenario implements the GAA (Generative Adversarial Agents) system for evaluating medical dialogue agents through doctor-patient interactions.

## Overview

The system evaluates a doctor agent's ability to persuade patients to accept surgical treatment across diverse patient personas (16 MBTI personality types × 2 genders × 2 medical conditions = 64 combinations).

## Key Features

### Information Asymmetry Design

A critical feature that mirrors real medical practice:

- **Doctor Agent Receives (Visible Information):**
  - Age, gender, medical condition
  - Diagnosis and symptoms
  - Recommended surgical treatment
  - Clinical facts about the case

- **Patient Agent Uses (Hidden from Doctor):**
  - Full system prompt with MBTI personality traits
  - Dynamically generated background story
  - Personality-driven concerns and fears
  - Behavioral patterns and communication style

**Rationale:** This tests the doctor's ability to discover patient personality through dialogue observation and adapt their communication style in real-time, just like in actual medical practice.

### Round-Based Evaluation

Each dialogue consists of multiple rounds:

1. **Doctor** sends response (addressing concerns, presenting evidence)
2. **Patient** generates personality-driven response
3. **Judge** evaluates the round:
   - Scores: Empathy (0-10), Persuasion (0-10), Safety (0-10)
   - Checks stop conditions (patient left/accepted/max rounds)
   - Continues or generates final comprehensive report

## Architecture

This scenario integrates with the `agentbeats` infrastructure using the standard `EvalRequest` and `EvalResult` models for compatibility.

### Green Agents (Evaluators)
- **Judge Agent**: Central orchestrator managing round-by-round evaluation
- **Patient Agent**: Simulates patient with personality-driven behavior
- **Patient Constructor**: Generates patient personas from MBTI/gender/case templates
- **Per-Round Scoring Engine**: LLM-based evaluation of each round
- **Stop Condition Detector**: Determines dialogue termination
- **Report Generator**: Creates comprehensive performance reports

### Purple Agents (Evaluated)
- **Doctor Agent**: Agent being evaluated (example implementation provided)

## Quick Start

### 1. Set up environment

```bash
# Copy sample.env to .env and configure
cp sample.env .env

# Edit .env with your API credentials
# API_KEY=your_api_key_here
# BASE_URL=your_api_endpoint
# DEFAULT_MODEL=gpt-4
```

### 2. Run evaluation

```bash
# Evaluate single persona (fast testing)
agentbeats-run scenarios/medical_dialogue/scenario.toml

# Edit scenario.toml to evaluate multiple personas
# persona_ids = ["INTJ_M_PNEUMO", "ESFP_F_LUNG"]

# Evaluate all 64 personas (comprehensive evaluation)
# persona_ids = ["all"]
```

## Configuration

### Persona ID Format

`{MBTI}_{GENDER}_{CASE}`

**MBTI Types (16):**
- Analysts: INTJ, INTP, ENTJ, ENTP
- Diplomats: INFJ, INFP, ENFJ, ENFP
- Sentinels: ISTJ, ISFJ, ESTJ, ESFJ
- Explorers: ISTP, ISFP, ESTP, ESFP

**Gender:**
- M (male)
- F (female)

**Medical Cases:**
- PNEUMO (pneumothorax)
- LUNG (lung_cancer)

**Examples:**
- `INTJ_M_PNEUMO`: Male INTJ with pneumothorax
- `ESFP_F_LUNG`: Female ESFP with lung cancer
- `all`: All 64 combinations

### Scenario Configuration (scenario.toml)

```toml
[config]
persona_ids = ["INTJ_M_PNEUMO"]  # Single persona
# persona_ids = ["all"]  # All 64 personas
max_rounds = 5  # Maximum dialogue rounds
```

## Evaluation Metrics

### Per-Round Scores (0-10 each)

1. **Empathy Score**
   - Emotional tone and warmth
   - Acknowledgment of patient concerns
   - Rapport-building effectiveness

2. **Persuasion Score**
   - Impact on patient receptiveness
   - Argument quality and adaptation
   - Progress toward decision

3. **Safety Score**
   - Medical accuracy
   - Informed consent
   - Appropriate safety recommendations

### Final Report Includes

- Aggregate scores (mean across all rounds)
- Overall performance (0-100 weighted score)
- Strengths and weaknesses analysis
- Key dialogue moments
- Actionable improvement recommendations
- Alternative approaches suggested

## Prompt Templates

Located in `prompts/` directory:

- `mbti/`: 16 MBTI personality descriptions
- `gender/`: Male and female context
- `cases/`: Pneumothorax and lung cancer case details

These text files are combined by the Patient Constructor to generate coherent patient personas.

## Developing Custom Doctor Agents

Replace `purple_agents/doctor_agent.py` with your implementation:

### Requirements

1. **Implement A2A Protocol**: Use A2A SDK or Google ADK
2. **Accept Context Messages**: Receive PatientClinicalInfo + dialogue history
3. **Generate Responses**: Return text message with doctor's response
4. **Maintain Conversation**: Track context across rounds

### Example Implementation Patterns

**Using Google ADK** (see `purple_agents/doctor_agent.py`):
```python
from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

root_agent = Agent(
    name="doctor",
    model=model_config,
    description="Medical doctor agent",
    instruction="Your medical consultation instructions..."
)

a2a_app = to_a2a(root_agent, agent_card=agent_card)
uvicorn.run(a2a_app, host=args.host, port=args.port)
```

**Custom A2A Implementation**: Follow patterns in `scenarios/debate/debater.py`

## File Structure

```
scenarios/medical_dialogue/
├── green_agents/
│   ├── __init__.py
│   ├── judge.py                    # Main orchestrator
│   ├── patient_agent.py            # Patient simulator
│   ├── patient_constructor.py      # Persona generator
│   ├── per_round_scoring.py        # Round evaluation
│   ├── stop_detector.py            # Stop condition detection
│   ├── report_generator.py         # Final report generation
│   ├── persona_manager.py          # Prompt template manager
│   └── common.py                   # Data models
├── purple_agents/
│   ├── __init__.py
│   └── doctor_agent.py             # Example doctor agent
├── prompts/
│   ├── mbti/                       # 16 personality types
│   ├── gender/                     # 2 gender contexts
│   └── cases/                      # 2 medical cases
├── scenario.toml                   # Configuration
└── README.md                       # This file
```

## Extending the System

### Adding New Medical Cases

1. Create new case file in `prompts/cases/`
2. Update `MEDICAL_CASES` in `persona_manager.py`
3. Add case code to persona ID format

### Adding New Personality Types

1. Add personality description in `prompts/mbti/` (or new folder)
2. Update `PersonaManager` to support new type
3. Update persona ID format accordingly

### Customizing Evaluation Criteria

Modify scoring prompts in:
- `per_round_scoring.py`: Adjust scoring criteria
- `report_generator.py`: Customize report structure
- `stop_detector.py`: Change stop condition logic

## Troubleshooting

### Common Issues

**"Prompt file not found"**
- Ensure all prompt files exist in `prompts/` directories
- Check persona_id format matches file names

**"Invalid persona_id format"**
- Use format: `{MBTI}_{GENDER}_{CASE}`
- Example: `INTJ_M_PNEUMO` not `intj-male-pneumothorax`

**"Agent connection timeout"**
- Ensure both judge and doctor agents are running
- Check ports in scenario.toml match agent startup

**"Model API error"**
- Verify API_KEY and BASE_URL in .env
- Check model name is correct for your provider
- Ensure sufficient API quota/credits

## Performance Notes

- **Single persona**: ~2-5 minutes (5 rounds)
- **10 personas**: ~20-50 minutes
- **All 64 personas**: ~2-5 hours (depends on API speed)

Use `persona_ids = ["all"]` for comprehensive benchmarking, or select specific personas for targeted evaluation.

## Citation

If you use this evaluation framework in your research, please cite:

```
GAA: Generative Adversarial Agents for Safe Medical Dialogue Evaluation
OSEC Project, 2024
```

## License

See LICENSE file in repository root.
