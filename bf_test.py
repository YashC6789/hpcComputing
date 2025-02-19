import os
import torch
import argparse
import time
import subprocess
from accelerate import PartialState
from diffusers import DiffusionPipeline

# Load the model
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, cache_dir="/home/hice1/ychauhan9/scratch")
distributed_state = PartialState()
pipe.to(distributed_state.device)

with distributed_state.split_between_processes(["dog, 8K", "tiger, 8K"]) as prompt:
    image = pipe(prompt).images[0]
    print(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
    output_filename = f"generated_image_{distributed_state.process_index}.png"
    image_path = os.path.join("/home/hice1/ychauhan9/ondemand/data/sys/myjobs/projects/default/1/workingoutputs", output_filename)
    image.save(image_path)