from semchunk.semchunk import chunkerify

CHUNK_SIZE = 100


def getChunks(text: str):
    chunker = chunkerify(lambda text: len(text.split()), CHUNK_SIZE)
    chunks, offsets = chunker(text, offsets=True, overlap=0.3)
    return chunks
