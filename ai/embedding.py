from PIL import Image
from Models import nomic_text
from Models import git
# from Models import blip2
from helpers.helpers import load_imgs, load_texts
from sklearn.metrics.pairwise import cosine_similarity

imgs, im_fnames = load_imgs(r"ai/files/img_files")

texts, txt_fnames = load_texts(r"ai/files/txt_files")


txt_embeds = nomic_text.text_embedder(texts)
img_captions = git.caption(imgs)
print(img_captions)

img_embeds = nomic_text.text_embedder(img_captions)
test_embed = nomic_text.text_embedder(["an image of a man and a woman"])
sim = cosine_similarity(img_embeds[1].reshape(  # this takes the photo of a man and a woman
    1, -1), test_embed[0].reshape(1, -1))

print(f"Similarity b/w text, image embed given is:{sim}")
