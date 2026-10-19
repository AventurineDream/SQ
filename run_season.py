import logging
from src.engine import SimulationEngine
from src.inference import OllamaAdapter
from src.evals import AutoJudge
from src.rules import Skill

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    print("[*] Initializing Simulation Query: Season 2")
    
    # 1. Load the Worldbook
    wb = {
        "characters": {
            "Sage": {"hp": 18, "skills": {Skill.ATHLETICS: 3}, "inventory": ["Pulse Rifle", "Combat Knife"]},
            "Rory": {"hp": 12, "skills": {Skill.TECH: 4}, "inventory": ["Plasma Torch"]},
            "Kaelen": {"hp": 14, "skills": {Skill.PERCEPTION: 3}, "inventory": ["Medkit"]},
            "Jace": {"hp": 13, "skills": {Skill.ACROBATICS: 3}, "inventory": ["Sidearm"]}
        }
    }
    
    # 2. Spin up the Engine
    engine = SimulationEngine(season_id="s2-scifi-horror", worldbook_data=wb)
    
    # 3. Inject Real Ollama Adapters (Make sure `ollama serve` is running!)
    engine.orchestrator_model = OllamaAdapter(model_name="qwen2.5:32b-instruct-q4_K_M") # Or whatever your 30B is named
    engine.actors = {
        "Sage": OllamaAdapter(model_name="gpt-oss:20b"), 
        "Rory": OllamaAdapter(model_name="qwen2.5:7b"),
        "Kaelen": OllamaAdapter(model_name="llama3.1:8b"),
        "Jace": OllamaAdapter(model_name="mistral:7b")
    }
    
    # 4. Attach the Auto-Judge
    mistral_judge = OllamaAdapter(model_name="mistral-small3.2:24b")
    engine.auto_judge = AutoJudge(db=engine.db, judge_model=mistral_judge)
    
    # 5. Execute!
    print("[*] Commencing Day 1...")
    engine.run_day(day=1)
    
    print("[*] Day 1 Complete. Check data/sq_ledger.db for telemetry.")

if __name__ == "__main__":
    main()