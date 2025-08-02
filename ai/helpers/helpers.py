from PIL import Image, ImageFile
import os
import datetime
import numpy as np
from typing import Tuple


def load_imgs(dir_path: str) -> Tuple[list[ImageFile.ImageFile], list[str], list[dict]]:
    images, fnames, metadatas = [], [], []
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith((".jpg", ".png", ".jpeg")):
            fpath = os.path.join(dir_path, fname)
            img = Image.open(fpath)
            images.append(img)

            raw_metadata = os.stat(fpath)
            metadata = {
                "created_at": datetime.datetime.fromtimestamp(raw_metadata.st_ctime).strftime("%A, %B %d, %Y %I:%M:%S"),
                "last_modified": datetime.datetime.fromtimestamp(raw_metadata.st_mtime).strftime("%A, %B %d, %Y %I:%M:%S"),
                "file_size": raw_metadata.st_size/1024,
            }
            metadatas.append(metadata)

    print(f"Loaded {len(images)} images.")
    return images, fnames, metadatas


def load_texts(folder: str) -> Tuple[list[str], list[str], list[dict]]:
    # Task Instruction required for nomic embed model
    nomic_task_instruction = "clustering: "

    texts, filenames, metadatas = [], [], []

    for fname in sorted(os.listdir(folder)):
        if fname.endswith((".txt", ".md")):
            fpath = os.path.join(folder, fname)

            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if not text:  # Checks if text isn't just empty
                    pass

                raw_metadata = os.stat(fpath)
                metadata = {
                    "created_at": datetime.datetime.fromtimestamp(raw_metadata.st_ctime).strftime("%A, %B %d, %Y %I:%M:%S"),
                    "last_modified": datetime.datetime.fromtimestamp(raw_metadata.st_mtime).strftime("%A, %B %d, %Y %I:%M:%S"),
                    "file_size": raw_metadata.st_size/1024,
                }

                text = nomic_task_instruction + text
                texts.append(text)
                filenames.append(fname)
                metadatas.append(metadata)

    print(f"Loaded {len(texts)} texts.")
    return texts, filenames, metadatas


def form_cluster_embed_map(embeddings: list[np.ndarray], cluster_ids: np.ndarray) -> dict[int, list[int]]:
    cluster_embed_map = {}  # stores cluster_id -> embeds mapping

    for idx, cluster_id in enumerate(cluster_ids):
        emb = embeddings[idx]

        if (int(cluster_id) not in cluster_embed_map.keys()):
            cluster_embed_map[int(cluster_id)] = []

        cluster_embed_map[cluster_id].append(emb)

    return cluster_embed_map
