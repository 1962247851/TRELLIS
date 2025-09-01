import torch
from safetensors.torch import save_file
import json
import os
# Load the config
config_path = "/home/public/PycharmProjects/TRELLIS/configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json"
with open(config_path, 'r') as f:
    config = json.load(f)
# Load the EMA checkpoint
ckpt_path = "/home/public/PycharmProjects/TRELLIS/outputs/metafood3d_finetuned/ckpts/denoiser_ema0.9999_step0120000.pt"
state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
# Save as safetensors
output_path = "/home/public/PycharmProjects/TRELLIS/test/convert/slat_flow_img_dit_L_64l8p2_fp16.safetensors"
save_file(state_dict, output_path)
# The config.json should already exist, but if you need to create it:
config_output = {
    "name": "ElasticSLatFlowModel",  # or whatever your decoder class name is
    "args": config['models']['denoiser']['args']
}
with open("/home/public/PycharmProjects/TRELLIS/test/convert/slat_flow_img_dit_L_64l8p2_fp16.json", 'w') as f:
    json.dump(config_output, f, indent=4)