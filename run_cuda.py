import subprocess
import os

current_dir = os.getcwd()
cuda_program_file = "sample_kernel.cu"
cuda_program_path = os.path.join(current_dir, cuda_program_file)

subprocess.run(["nvcc", cuda_program_path, "-o", "add_cuda"])
subprocess.run(["./add_cuda"], capture_output=True)