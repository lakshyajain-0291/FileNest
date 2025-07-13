from vllm import LLM
from transformers import AutoProcessor, AutoModelForVision2Seq
import time

print("Initializing GIT")
t1 = time.time()
# processor = AutoProcessor.from_pretrained("microsoft/git-base-textcaps")
# img_model = AutoModelForVision2Seq.from_pretrained(
#     "microsoft/git-base-textcaps").to("cuda")
processor = AutoProcessor.from_pretrained("microsoft/git-large-textcaps")
img_model = AutoModelForVision2Seq.from_pretrained(
    "microsoft/git-large-textcaps").to("cuda:0")

print(f"Finished initializing GIT in {time.time()-t1}s")


def caption(imgs):

    t1 = time.time()
    inputs = processor(images=imgs, return_tensors="pt",
                       use_fast=True).to("cuda:0")  # type: ignore
    out = img_model.generate(**inputs)  # type: ignore
    captions = []
    for i in range(len(out)):
        captions.append(processor.decode(  # type: ignore
            out[i], skip_special_tokens=True))

    print(f"Finished image captioning in {time.time()-t1}s")
    return captions
