import os
import torch
import argparse
import time
import tracemalloc
import subprocess
from accelerate import PartialState
from diffusers import DiffusionPipeline

# Load the model
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, cache_dir="/home/hice1/ychauhan9/scratch")
#distributed_state = PartialState()
#pipe.to(distributed_state.device)
pipe.to("cpu")

sTime = time.time()
tracemalloc.start()
for i, prompt in enumerate(["dog, 8K"]):
    image = pipe(prompt, num_inference_steps=20).images[0]
    #print(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    tracemalloc.stop()
    output_filename = f"GPU:1_image_{i}.png"
    image_path = os.path.join("/home/hice1/ychauhan9/ondemand/data/sys/myjobs/projects/default/1/workingoutputs", output_filename)
    image.save(image_path)

#with distributed_state.split_between_processes(["dog, 8K", "tiger, 8K", "shark, 8K", "dragon, 8k", "phoenix bird, 8K", "unicorn, 8K"]) as prompt:
#    images = pipe(prompt).images
#    for i, image in enumerate(images):
#        output_filename = f"generated_image_{distributed_state.process_index}_{i}.png"
#        image_path = os.path.join("/home/hice1/ychauhan9/ondemand/data/sys/myjobs/projects/default/1/workingoutputs", output_filename)
#        image.save(image_path)
eTime = time.time()
#print(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
print("Execution Time: " + str(eTime - sTime))