import torch

def generate_position_embeddings(n, x_start=0.0, x_end=1.0):
    """
    Divides [x_start, x_end] into n equal parts, picks a random x 
    in each, and returns a (n, 2) tensor of [cos(x), sin(x)].
    """
    # 1. Calculate interval width
    width = (x_end - x_start) / n
    
    # 2. Create the lower bounds for each of the n intervals
    # shape: (n,) -> [0, w, 2w, ..., (n-1)w]
    lower_bounds = torch.linspace(x_start, x_end - width, n)
    
    # 3. Select a random x within each interval [lower, lower + width]
    # torch.rand(n) gives n values in [0, 1)
    random_offsets = torch.rand(n) * width
    x_samples = lower_bounds + random_offsets
    
    # 4. Define the functions using PyTorch operations
    def cos_func(x):
        return 0.2 * torch.cos(torch.pi * (2 * x + 1))

    def sin_func(x):
        return 0.2 * torch.sin(torch.pi * (2.5 * x + 2/3))

    # 5. Compute values and stack into (n, 2) tensor
    cos_vals = cos_func(x_samples)
    sin_vals = sin_func(x_samples)
    
    # Stack along dimension 1 to get (n, 2)
    return torch.stack((cos_vals, sin_vals), dim=1)
