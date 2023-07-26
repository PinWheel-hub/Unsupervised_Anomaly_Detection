import torch

a = torch.tensor([[[[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]], 
                    [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
                    [[2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]],
                    [[2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]]]])

a = a.permute((1, 2, 0, 3)).unsqueeze(0)
a = a.unsqueeze(2)
a = a.reshape((2, 2, 2, 2, -1, 3))
a = a.permute((0, 2, 1, 3, 4, 5)).reshape((2, 2, -1, 3))
print(a, a.shape)
print(a[1, 0, :], a.shape)
print(a.reshape((4, -1))[2, :], a.shape)