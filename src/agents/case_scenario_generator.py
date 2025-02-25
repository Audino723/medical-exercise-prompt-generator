from src.agents.base_llm_model import BaseLLMModel
from typing import Dict, List

class CaseScenarioGenerator(BaseLLMModel):
    def generate_prompt(self, extracted_data: Dict[str, str]) -> str:
        """Constructs a system prompt for generating case-based scenarios."""
        return f"""
        You are an AI medical assistant designed to evaluate clinical decisions.  
        Your task is to analyze a given **medical case scenario** and evaluate different possible actions a student might take.  
        
        For each action:
        1. **Determine if it is medically correct or incorrect.**  
        2. **Provide a short but precise medical explanation.**  
        
        ### Example Input:  
        {{
          "context": "{extracted_data['context']}",
          "question": "{extracted_data['question']}",
          "expected_answer": "{extracted_data['expected_answer']}"
        }}
        
        ### Expected Output:  
        {{
          "context": "{extracted_data['context']}",
          "question": "{extracted_data['question']}",
          "actions": {{
            "A": {{
              "response": "Yes, because oxygen levels are within range.",
              "correctness": "Incorrect",
              "explanation": "Oxygen levels alone do not determine ventilation adequacy. Elevated PaCO₂ indicates hypoventilation."
            }},
            "B": {{
              "response": "No, because the high PaCO₂ indicates poor ventilation.",
              "correctness": "Correct",
              "explanation": "Correct! High PaCO₂ suggests hypoventilation, meaning ventilation is inadequate."
            }},
            "C": {{
              "response": "It cannot be determined without additional tests.",
              "correctness": "Incorrect",
              "explanation": "PaCO₂ levels are sufficient to determine ventilation adequacy without additional tests."
            }}
          }}
        }}
        
        Now, evaluate the following medical case and generate structured responses:
        """
    
    def generate_scenarios(self, extracted_data: Dict[str, str]) -> Dict:
        """Uses LLM to generate case-based scenarios and evaluate responses."""
        prompt = self.generate_prompt(extracted_data)
        return self.generate_completion(prompt)
    
if __name__ == "__main__":
    extracted_data = {
        "context": "A 20-year-old student was admitted after an overdose on sedatives. His blood gas readings were: PaO₂ = 65 mm Hg, PaCO₂ = 60 mm Hg.",
        "question": "Is alveolar ventilation of the patient adequate?",
        "expected_answer": "No, ventilation is inadequate because the high PaCO₂ indicates poor ventilation."
    }
    
    agent = CaseScenarioGenerator(model="gpt-4")
    case_scenarios = agent.generate_scenarios(extracted_data)
    print(case_scenarios)
