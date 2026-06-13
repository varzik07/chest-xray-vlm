"""Quick sanity check: is the GPU visible and how much VRAM is free?
Run:  python src/check_gpu.py
"""
import torch


def main() -> None:
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No CUDA GPU detected — training will be very slow on CPU.")
        return

    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    total = props.total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(i) / 1024**3
    allocated = torch.cuda.memory_allocated(i) / 1024**3
    print(f"GPU             : {props.name}")
    print(f"Total VRAM      : {total:.1f} GB")
    print(f"Reserved/Alloc  : {reserved:.2f} / {allocated:.2f} GB")
    print(f"Free (approx)   : {total - reserved:.1f} GB")

    # Tiny op to confirm compute works end to end
    x = torch.randn(1000, 1000, device="cuda")
    y = (x @ x).sum().item()
    print(f"Matmul on GPU OK: {y:.1f}")


if __name__ == "__main__":
    main()
