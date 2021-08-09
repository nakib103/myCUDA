import subprocess

cuda_program = "sample_kernel.cu"
subprocess.run(["nvcc", cuda_program, "-o", "add_cuda"])
subprocess.run(["./add_cuda"], capture_output=True)