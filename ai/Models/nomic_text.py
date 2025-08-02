import time
from llama_cpp import Llama
import numpy as np
# print(llama_cpp.llama_supports_gpu_offload())

text_model = Llama.from_pretrained(
    repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
    filename="nomic-embed-text-v1.5.Q8_0.gguf",
    embedding=True,
    verbose=False,
    n_gpu_layers=-1)


def text_embedder(texts: list[str]) -> list[np.ndarray]:
    t1 = time.time()

    embeds = []
    for text in texts:
        emb = text_model.create_embedding(text)
        emb = emb["data"][0]["embedding"]
        emb = np.array(emb)
        embeds.append(emb)

    print(f"Text embeds generated in {time.time() - t1}s")
    return embeds
