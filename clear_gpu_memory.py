"""
GPU belleğini temizleme scripti
"""
import torch
import gc

def clear_gpu_memory():
    """GPU belleğini temizle"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        print("GPU belleği temizlendi")
    else:
        print("CUDA mevcut değil")

if __name__ == '__main__':
    clear_gpu_memory()


