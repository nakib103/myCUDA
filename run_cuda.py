import subprocess
import os

current_dir = os.getcwd()
cuda_program_file = "sample_kernel.cu"
cuda_program_path = os.path.join(current_dir, cuda_program_file)

object_file = "add_cuda"
object_file_path = os.path.join(current_dir, object_file)

subprocess.run(["nvcc", cuda_program_path, "-o", object_file_path])
subprocess.run([object_file_path], capture_output=True)