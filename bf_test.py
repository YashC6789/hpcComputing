import os
import torch
import argparse
import time
from diffusers import DiffusionPipeline

# Parse GPU count argument
parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("--rand", type=int, default=0, help="Random Num for Output")
args = parser.parse_args()

# Load the model
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.bfloat16, cache_dir="/home/hice1/ychauhan9/scratch")
pipe.unet.to("cuda:0")         # UNet to GPU 0
pipe.text_encoder.to("cuda:1")  # Text encoder to GPU 1
pipe.vae.to("cuda:2")          # VAE to GPU 0
pipe.to("cuda:0")              # Keep pipeline on main GPU

# Define prompt
prompt = "Dragon breathing fire, dark color palette, detailed, 8k"

start_time = time.time()
image = pipe(prompt).images[0]
end_time = time.time()

output_filename = f"generated_image_{args.rand}.png"

# Define image path in the existing scratch folder
image_path = os.path.join("/home/hice1/ychauhan9/ondemand/data/sys/myjobs/projects/default/1/workingoutputs", output_filename)

# Save image in scratch folder
image.save(image_path)

# Print benchmark results
elapsed_time = end_time - start_time
print(f"GPU(s) Used: {args.gpus} | Time Taken: {elapsed_time:.2f} seconds")

# Save benchmark results to a file
with open(f"/home/hice1/ychauhan9/ondemand/data/sys/myjobs/projects/default/1/workingoutputs/benchmark_results_{args.rand}.txt", "w") as f:
    f.write(f"GPU(s) Used: {args.gpus}\n")
    f.write(f"Time Taken: {elapsed_time:.2f} seconds\n")