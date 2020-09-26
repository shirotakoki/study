import torch

a = torch.zeros([2, 4], dtype=torch.int32)
print(a)
print(a[0][0].item())
print(a.numpy())


res = torch.tensor(0.1, dtype=torch.float)
torch.add(res, 0.3, out=res)
print(res)
