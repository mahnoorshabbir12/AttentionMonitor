import torch
from ultralytics import YOLO

def main():
    model = YOLO("models/exp.pt")
    print(f"Task: {model.task}")
    print(f"Names: {model.names}")
    
    # Check if weights are zero or normal
    state_dict = model.model.state_dict()
    first_key = list(state_dict.keys())[0]
    tensor = state_dict[first_key]
    print(f"First layer ({first_key}) shape: {tensor.shape}")
    print(f"First layer mean: {tensor.float().mean().item():.6f}")
    print(f"First layer std: {tensor.float().std().item():.6f}")
    
    # Are there any NaN values in the weights?
    has_nan = any(torch.isnan(v).any() for v in state_dict.values() if v.is_floating_point())
    print(f"Contains NaN weights: {has_nan}")

if __name__ == "__main__":
    main()
