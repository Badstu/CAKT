import torch
t = torch.tensor([[1, 2, 2, 3], [1, 3, 4, 2]])
values, indices = t.topk(2, 1, True)
print(values, '\n', indices)
t1 = torch.zeros([2, 4])
for i, m in enumerate(indices):
    # for j in range(2):
    t1[i, m] = 1
print(t1)



# for i in range(t.shape[0]):
#     x = torch.zeros(5)
#     y = torch.zeros(5)
#     idx = list((t[i, :]==2).nonzero())
#
#
# print(index)
# index = list(idx[0])
# print(int(index[0]))
# print(t.select(list(idx[0]).))
# for i in range(t.shape[0]):
#     idx = list((t[i,:]==2).nonzero())
#     idx = torch.cat([torch.LongTensor(idx[i]) for i in range(len(idx))], 0)
#     index.append(idx)
#
# print(index)
# # x = torch.randn(3, 4)
#
# # indices = torch.LongTensor([0, 2, 3])
# for i in range(t1.shape[0]):
#     print(torch.index_select(t1[i, :], 0, index[i]))


