from typing import Tuple
import pymupdf


def getTextImages(path: str) -> Tuple[list[str], list]:
    texts, images = [], []
    doc = pymupdf.open(path)  # open a document

    for page in doc:  # iterate the document pages
        text = page.get_textpage().extractText()  # get plain text (is in UTF-8)
        imgs = page.get_images()
        texts.append(text)
        images.append(imgs)

    return texts, images
