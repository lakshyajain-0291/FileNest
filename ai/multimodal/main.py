from PIL import Image
import time
import os
from sklearn.metrics.pairwise import cosine_similarity
import llama_cpp
# import blip_captioning
import nomic_text
import git
print(llama_cpp.llama_supports_gpu_offload())


def load_imgs(dir_path: str):
    images, fnames = [], []
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith((".jpg", ".png", ".jpeg")):
            fpath = os.path.join(dir_path, fname)
            img = Image.open(fpath)
            images.append(img)
    return images, fnames


imgs, im_fnames = load_imgs(r"./files/img_files")
print(f"Loaded {len(imgs)} images.")


def load_texts(folder):
    typee = "clustering: "
    texts, filenames = [], []

    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".txt"):
            fpath = os.path.join(folder, fname)

            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    text = typee + text
                    texts.append(text)
                    filenames.append(fname)
    return texts, filenames


texts, txt_fnames = load_texts(r"./files/txt_files")
print(f"Loaded {len(texts)} texts.")

txt_embeds = nomic_text.text_embedder(texts)
img_captions = git.caption(imgs)
print(img_captions)

img_embeds = nomic_text.text_embedder(img_captions)
test_embed = nomic_text.text_embedder(["an image of a man and a woman"])
print(cosine_similarity(img_embeds[1].reshape(1, -1),
                        test_embed[0].reshape(1, -1)))
