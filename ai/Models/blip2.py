import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import time

print("Initializing Blip2")

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
img_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map="auto")

t1 = time.time()
print(f"Finished initializing BLIP2 in {time.time()-t1}s")


def caption(imgs):
    t1 = time.time()
    inputs = processor(images=imgs, return_tensors="pt",
                       use_fast=True).to("cuda")  # type: ignore
    out = img_model.generate(**inputs)  # type: ignore
    captions = []
    for i in range(len(out)):
        captions.append(processor.decode(  # type: ignore
            out[i], skip_special_tokens=True))

    print(f"Finished image captioning in {time.time()-t1}s")
    return captions
