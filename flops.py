import torch
from torchprofile import profile_macs
from models.parcnet_v2 import parcnet_v2_b2

model = parcnet_v2_b2()
inputs = torch.randn(1, 3, 224, 224)

params = sum(p.numel() for p in model.parameters())
macs = profile_macs(model, inputs)
print("{:<30}  {:<20}".format("Computational complexity: ", macs))
print("{:<30}  {:<20}".format("Number of parameters: ", params))


# import torch
# from ptflops import get_model_complexity_info

# with torch.cuda.device(3):
#     macs, params = get_model_complexity_info(
#         model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True
#     )
#     print("{:<30}  {:<8}".format("Computational complexity: ", macs))
#     print("{:<30}  {:<8}".format("Number of parameters: ", params))
