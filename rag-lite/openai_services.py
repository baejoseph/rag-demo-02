from openai import OpenAI
from typing import List, Protocol, Dict

# === Embedding Service Protocol ===

class EmbeddingService(Protocol):
    def embed_text(self, text: str) -> List[float]:
        ...

# === Generation Service Protocol ===

class GenerationService(Protocol):
    def generate_response(self, augmented_prompt: str) -> str:
        ...

# === OpenAI Implementation ===

class OpenAIEmbeddingService:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text.")

        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

class OpenAIGenerationService:
    def __init__(self, api_key: str, model: str = "gpt-4.1-nano-2025-04-14", memory_window: int = 1):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.memory_window = memory_window
        self.chat_memory: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    def generate_response(self, augmented_prompt: str) -> str:
        if not augmented_prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        self.chat_memory.append({"role": "user", "content": augmented_prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_memory,
            temperature=0.7,
            max_tokens=500
        )

        assistant_reply = response.choices[0].message.content.strip()

        # Add only the last N lines of the assistant's response to memory
        reply_lines = assistant_reply.splitlines()
        truncated_reply = "\n".join(reply_lines[-self.memory_window:])

        self.chat_memory.append({"role": "assistant", "content": truncated_reply})

        return assistant_reply
