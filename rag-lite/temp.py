from ollama_services import OllamaEmbeddingService

embedding_service = OllamaEmbeddingService()

print(embedding_service.embed_text("The sky is blue because of rayleigh scattering"))