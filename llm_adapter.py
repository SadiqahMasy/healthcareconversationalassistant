import requests
import json
from typing import Dict, List, Optional, Union, Any

class OllamaAdapter:
    """
    Adapter for calling Ollama models running on a local machine.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3"):
        """
        Initialize the Ollama adapter.
        
        Args:
            base_url: The base URL of the Ollama API (default: http://localhost:11434)
            default_model: The default model to use (default: llama3)
        """
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        
    def generate(self, 
                prompt: str, 
                model: Optional[str] = None, 
                system_prompt: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: The prompt to generate a completion for
            model: The model to use (defaults to the instance's default_model)
            system_prompt: Optional system prompt to provide context
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum number of tokens to generate (default: None)
            stop_sequences: List of strings that will stop generation if encountered
            
        Returns:
            The generated text as a string
        """
        model = model or self.default_model
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            # The API returns one JSON object per line
            response_text = ""
            for line in response.text.strip().split('\n'):
                data = json.loads(line)
                if "response" in data:
                    response_text += data["response"]
                if data.get("done", False):
                    break
                    
            return response_text
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {e}")
    
    def chat(self,
            messages: List[Dict[str, str]],
            model: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            stop_sequences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a chat completion for the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     Roles can be 'system', 'user', or 'assistant'
            model: The model to use (defaults to the instance's default_model)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum number of tokens to generate (default: None)
            stop_sequences: List of strings that will stop generation if encountered
            
        Returns:
            Dictionary containing the response and other metadata
        """
        model = model or self.default_model
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        if stop_sequences:
            payload["stop"] = stop_sequences
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            # The API returns one JSON object per line with streaming
            response_text = ""
            for line in response.text.strip().split('\n'):
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    response_text += data["message"]["content"]
                    
            # Construct a response similar to what the API would return
            result = {
                "model": model,
                "message": {"role": "assistant", "content": response_text},
                "done": True
            }
            
            return result
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Ollama API: {e}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in the local Ollama instance.
        
        Returns:
            List of dictionaries containing model information
        """
        url = f"{self.base_url}/api/tags"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error listing Ollama models: {e}")

    def pull_model(self, model_name: str) -> None:
        """
        Pull a model from the Ollama library.
        
        Args:
            model_name: The name of the model to pull
        """
        url = f"{self.base_url}/api/pull"
        
        payload = {
            "name": model_name
        }
        
        try:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()
            
            # This is a long-running operation that streams progress
            for line in response.iter_lines():
                if line:
                    print(json.loads(line.decode('utf-8')))
                    
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error pulling Ollama model: {e}")

# Example usage
if __name__ == "__main__":
    ollama = OllamaAdapter()
    
    # Simple text generation
    response = ollama.generate(
        prompt="Explain quantum computing in simple terms",
        model="llama3.2:3b-instruct-q4_0",
        temperature=0.7
    )
    print(f"Generate Response:\n{response}\n")
    
    # Chat completion
    chat_response = ollama.chat(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the three laws of robotics?"}
        ],
        model="llama3.2:3b-instruct-q4_0"
    )
    print(f"Chat Response:\n{chat_response['message']['content']}\n")
    
    # List available models
    try:
        models = ollama.list_models()
        print(f"Available models: {[model['name'] for model in models]}")
    except Exception as e:
        print(f"Couldn't list models: {e}")
