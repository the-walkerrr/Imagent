import ollama
import config
from VLMResponseFormat import ImagePayload
from image_processing.preprocess import preprocess_image

def analyse_image(image_path: str) -> ImagePayload:
    client = ollama.Client(host=config.OLLAMA_HOST)
    image_bytes = preprocess_image(image_path)
    print(f"[imagent] Sending image to Ollama model: {config.OLLAMA_MODEL} @ {config.OLLAMA_HOST} …")
    response = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": config.ANALYSIS_PROMPT,
            "images": [image_path],
        }],
        format=ImagePayload.model_json_schema(),
        options={"num_ctx": config.OLLAMA_NUM_CTX},
    )
    return ImagePayload.model_validate_json(response.message.content)

def embed_search_text(search_text: str) -> list[float]:
    client = ollama.Client(host=config.OLLAMA_HOST)
    response = client.embed(
        model=config.OLLAMA_EMBEDDING_MODEL,
        input=[search_text],
    )
    return list(response.embeddings[0])
