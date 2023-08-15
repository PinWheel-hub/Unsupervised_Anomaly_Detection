import torch

a = torch.tensor([[[0, 10, 0], [0, 0, 0], [1, 1, 1], [20, 1, 1]], 
                    [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
                    [[2, 2, 2], [2, 2, 30], [3, 3, 3], [3, 3, 3]],
                    [[2, 2, 2], [2, 2, 2], [3, 40, 3], [3, 3, 3]]])
a = a.reshape((2, 2, 2, 2, 3))
a = a.permute((0, 2, 1, 3, 4)).reshape((2, 2, -1)).unsqueeze(-1)
print(a, a.shape)
print(a[1, 0, :])
b = torch.tensor([[[1, 2]]])
b = b.unsqueeze(2)
c = a @ b
print(c , c.shape)
index = torch.tensor([[1, 3], [5, 7]])
s = c[torch.arange(index.shape[0])[:, None, None],
                            torch.arange(index.shape[1])[None, :, None],
                            index[:, :, None], :]
s = s.expand_as(c)
squared_diff = (s - c)**2
distance = squared_diff.sum(dim=-1)
distance = torch.sqrt(distance)
print(distance, distance.shape)