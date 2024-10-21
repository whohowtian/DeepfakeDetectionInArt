#path dataset: /home/dani/datasets/wikiart
import torch
from PIL import Image
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tools import mask_top_half, compose_side_by_side,get_shuffled_file_paths,mask_random
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float32
)
pipe.to("cpu")

directory="C:/Users/cedri/Downloads/archive/similar/inpainting" ## this is the dataset path
list_files = get_shuffled_file_paths(directory)
count=20000
print(list_files[0])
for image_address in list_files:
    print(image_address)
    if not image_address.lower().endswith(".png"):
        continue
    image= Image.open(image_address).convert("RGB")
    width, height = image.size
    name="C:/Users/cedri/Downloads/PyTorch-GAN_2/PyTorch-GAN/implementations/cyclegan/diffuser_images/"+str(count) ## change path
    os.makedirs(name, exist_ok=True)
    masked_image=mask_random(image_address, output_path=name+"/mask.png").convert("RGB").resize((128, 128))
    prompt = "generate a painting compatible with the rest of the image"
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    image_inpainting = pipe(prompt=prompt, image=image.resize((128, 128)), mask_image=masked_image).images[0]
    compose_side_by_side(image_inpainting.resize((width, height)),image.resize((128, 128)).resize((width, height)),name+"/group.png")
    image_inpainting.resize((width, height)).save(name+"/inpainting.png")
    (image.resize((128, 128))).resize((width, height)).save(name+"/original.png")
    count+=1
    if(count>count+1500):
        break