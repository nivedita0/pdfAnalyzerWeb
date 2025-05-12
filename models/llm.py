import subprocess
import os
import json
from typing import Dict, Any, Optional

class LLMInterface:
    """Interface to different LLM backends"""
    
    def __init__(self, default_model="llama3", backend="ollama"):
        self.default_model = default_model
        self.backend = backend
        self.ollama_path = "C:/Users/Nivedita/AppData/Local/Programs/Ollama/ollama.exe"
        
        # Validate ollama path
        if backend == "ollama" and not os.path.exists(self.ollama_path):
            print(f"Warning: Ollama not found at {self.ollama_path}. Make sure it's installed.")
    
    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate text based on prompt using selected backend"""
        if model is None:
            model = self.default_model
            
        if self.backend == "ollama":
            return self._query_ollama(prompt, model)
        elif self.backend == "openrouter":
            return self._query_openrouter(prompt, model)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _query_ollama(self, prompt: str, model: str) -> str:
        """Send query to Ollama and get response"""
        command = [self.ollama_path, "run", model]
        try:
            # Use subprocess with proper encoding
            proc = subprocess.Popen(
                command, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                encoding='utf-8',
                errors='replace'
            )
            
            # Use ASCII-compatible prompt to avoid encoding issues
            safe_prompt = prompt.encode('ascii', 'ignore').decode('ascii')
            output, err = proc.communicate(input=safe_prompt)
            
            if err:
                print(f"Warning from Ollama: {err}")
                
            return output.strip()
            
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return f"Error: Could not get response from Ollama: {e}"
    
    def _query_openrouter(self, prompt: str, model: str) -> str:
        """Send query to OpenRouter API"""
        try:
            import openai
            
            # Configure OpenAI client for OpenRouter
            openai.api_key = os.getenv("OPENROUTER_API_KEY")
            if not openai.api_key:
                with open(".env") as f:
                    for line in f:
                        if line.startswith("OPENROUTER_API_KEY"):
                            openai.api_key = line.split("=")[1].strip()
                            break
                            
            openai.base_url = "https://openrouter.ai/api/v1"
            
            completion = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                extra_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "PDF Research Assistant",
                }
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error querying OpenRouter: {e}")
            return f"Error: Could not get response from OpenRouter: {e}"