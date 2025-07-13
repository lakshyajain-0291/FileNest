from PIL import Image
import os


def load_imgs(dir_path: str):
    images, fnames = [], []
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith((".jpg", ".png", ".jpeg")):
            fpath = os.path.join(dir_path, fname)
            img = Image.open(fpath)
            images.append(img)
    print(f"Loaded {len(images)} images.")
    return images, fnames


def load_texts(folder):
    typee = "clustering: "
    texts, filenames = [], []

    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".txt"):
            fpath = os.path.join(folder, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:  # Checks if text isnt just empty
                    text = typee + text
                    texts.append(text)
                    filenames.append(fname)
    print(f"Loaded {len(texts)} texts.")
    return texts, filenames
