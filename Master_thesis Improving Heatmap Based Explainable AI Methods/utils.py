import torch
import dnnlib
import legacy

def load_discriminator(network_pkl):
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f) 
    D = network['D'].to(device).eval()
    return D
