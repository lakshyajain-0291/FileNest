import time
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large")
img_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large").to("cuda")  # type: ignore

# processor = BlipProcessor.from_pretrained("unography/blip-large-long-cap")
# img_model = BlipForConditionalGeneration.from_pretrained(
#     "unography/blip-large-long-cap").to("cuda")  # type: ignore


def caption(imgs):
    print("Initializing Blip")

    t1 = time.time()
    print(f"Finished initializing BLIP in {time.time()-t1}s")

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
