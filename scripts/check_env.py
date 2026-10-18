"""Print what PyTorch sees of the GPU. Run inside the WSL venv."""
import torch

print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
print("GPU visible:", torch.cuda.is_available())

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"{name} | compute capability {major}.{minor} | {total_gb:.1f} GB VRAM")
    # sm_120 must appear here, otherwise this PyTorch build cannot run kernels on the card
    print("architectures in this build:", torch.cuda.get_arch_list())

    # Tiny matmul to prove a kernel really runs on the card (a few MB, negligible)
    x = torch.randn(256, 256, device="cuda")
    print("matmul on GPU OK, checksum:", float((x @ x).sum()))
