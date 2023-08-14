import torch

a = torch.tensor([[[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]], 
                    [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
                    [[2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]],
                    [[2, 2, 2], [2, 2, 2], [3, 3, 3], [3, 3, 3]]])
a = a.reshape((2, 2, 2, 2, 3))
a = a.permute((0, 2, 1, 3, 4)).reshape((2, 2, -1)).unsqueeze(-1)
print(a, a.shape)
print(a[1, 0, :], a.shape)
b = torch.tensor([[[1, 2]]])
b = b.unsqueeze(2)
c = a @ b
print(c , c.shape)