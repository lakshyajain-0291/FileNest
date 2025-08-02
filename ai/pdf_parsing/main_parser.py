import extract
import chunker

path = r'ai/files/pdf_files/Attention.pdf'
texts, images = extract.getTextImages(path)

chunks = chunker.getChunks(str(texts))
for chunk in chunks[:5]:
    print(str(chunk))
