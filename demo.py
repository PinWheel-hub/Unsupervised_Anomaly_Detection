import torch

# a = torch.tensor([[[0, 10, 0], [0, 0, 0], [1, 1, 1], [20, 1, 1]], 
#                     [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
#                     [[2, 2, 2], [2, 2, 30], [3, 3, 3], [3, 3, 3]],
#                     [[2, 2, 2], [2, 2, 2], [3, 40, 3], [3, 3, 3]]])
# a = a.reshape((2, 2, 2, 2, 3))
# a = a.permute((0, 2, 1, 3, 4)).reshape((2, 2, -1)).unsqueeze(-1)
# print(a, a.shape)

# a = a.reshape((2, 2, 2, 2, -1))
# a = a.permute((0, 2, 1, 3, 4)).reshape((-1, 3))
# print(a, a.shape)

# b = torch.tensor([[[1, 2]]])
# b = b.unsqueeze(2)
# c = a @ b
# print(c , c.shape)
# index = torch.tensor([[1, 3], [5, 7]])
# s = c[torch.arange(index.shape[0])[:, None, None],
#                             torch.arange(index.shape[1])[None, :, None],
#                             index[:, :, None], :]
# s1 = s.expand_as(c)
# squared_diff = (s1 - c)**2
# distance = squared_diff.sum(dim=-1)
# distance = torch.sqrt(distance)
# # print(distance, distance.shape)

# c[torch.arange(index.shape[0])[:, None, None],
#             torch.arange(index.shape[1])[None, :, None],
#             index[:, :, None]] = 0
# print(c)

def shift(matrix: torch.tensor, r: int, c: int):
    shifted_matrix = torch.zeros_like(matrix)
    if r > 0:
        new_rows = torch.arange(r, matrix.shape[0])
    else:
        new_rows = torch.arange(matrix.shape[0] + r)
    if c > 0:
        new_columns = torch.arange(c, matrix.shape[1])
    else:
        new_columns = torch.arange(matrix.shape[1] + c)
    if r <= 0 and c <= 0:
        shifted_matrix[new_rows[:, None], new_columns[None, :]] = matrix[-r:, -c:]
    elif r <= 0:
        shifted_matrix[new_rows[:, None], new_columns[None, :]] = matrix[-r:, :-c]
    elif c <= 0:
        shifted_matrix[new_rows[:, None], new_columns[None, :]] = matrix[:-r, -c:]
    else:
        shifted_matrix[new_rows[:, None], new_columns[None, :]] = matrix[:-r, :-c]
    return shifted_matrix

# Create a 4x4 tensor matrix
matrix = torch.arange(4 * 4 * 2 * 2).reshape(4, 4, 2, 2)
# Define the number of positions to shift
for i in (0, -1, 1):
    for j in (0, -1, 1):
        s = shift(matrix, i, j)
        print(s)