import time
import torch
import tracemalloc
from diffusers import DiffusionPipeline

# Function to print memory stats
def print_mps_memory():
    allocated = torch.mps.current_allocated_memory() / (1024 ** 2)  # Convert to MB
    print(f"Allocated Memory: {allocated:.2f} MB")

# Fix MPS memory issues
torch.backends.cuda.max_split_size_mb = 128

torch.mps.empty_cache()

# Load model
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", resume_download=True,torch_dtype=torch.float32)

# Optimize for MPS
print_mps_memory()
#pipe.to("mps", torch.float32)
#pipe.enable_attention_slicing()

# Run inference
print_mps_memory()
torch.mps.empty_cache()
startT = time.time()
tracemalloc.start()
prompt = "dog, 8K"
image = pipe(prompt, height=504, width=504, num_images_per_prompt=1).images[0]
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
tracemalloc.stop()
endT = time.time()
print("Execution Time: " + str(startT - endT))
# Show image
image.save("/Users/yashc/Documents/VIP/my_image2.png")

# Clear cache after generation
print_mps_memory()
torch.mps.empty_cache()
