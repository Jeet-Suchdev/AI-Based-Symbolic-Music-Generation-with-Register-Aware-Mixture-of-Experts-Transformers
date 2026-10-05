import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def device_report(device):
    report = {"device": str(device)}
    if device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["gpu_memory_gb"] = round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2)
        report["bf16"] = torch.cuda.is_bf16_supported()
    return report
