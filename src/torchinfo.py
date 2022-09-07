import torch
import torchvision

print('torch version:', torch.__version__)
print('torchvision version:', torchvision.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda version:', torch.version.cuda)
print('cuda arch list:', torch.cuda.get_arch_list())
deviceId = torch.cuda.current_device()
print('device:', torch.cuda.get_device_name(deviceId))
