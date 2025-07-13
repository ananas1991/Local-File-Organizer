import ollama

class OllamaTextInference:
    def __init__(self, model: str = "llama3") -> None:
        self.model = model

    def create_completion(self, prompt: str):
        """Return a dict mimicking OpenAI completion structure."""
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return {"choices": [{"text": response["message"]["content"]}]}


class OllamaVLMInference:
    def __init__(self, model: str = "llava") -> None:
        self.model = model

    def chat(self, prompt: str, image_path: str) -> str:
        """Return the model response for a prompt with an image."""
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
        )
        return response["message"]["content"]
