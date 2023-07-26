import torch
import paddle
from tqdm import tqdm

if __name__ == '__main__':

    dst_state_dict = torch.load('/home/wubw/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
    src_state_dict = paddle.load('/home/wubw/.cache/paddle/hapi/weights/resnet18.pdparams')

    state_tuples = {''}

    # print(dst_state_dict.keys())
    # print(src_state_dict.keys())
    
    src = {'weight': [], 'bias': [], 'mean': [], 'var': []}
    titles = list(src.keys())

    for k, v in tqdm(src_state_dict.items()):

        flag = False
        for t in titles:
            if t in k:
                src[t].append((k, v.numpy()))
                flag = True
                break
        if not flag:
            print(k)

    for k, v in src.items():
        print(k, len(v))

    print(len(src_state_dict))

    for k, v in tqdm(dst_state_dict.items()):
        if 'conv.bias' in k or 'num_batches_tracked' in k:
            continue
        flag = False
        for t in titles:
            if t in k:
                assert len(src[t]), '\033[1;35m {} empty list. \033[0m'.format(k)
                name, arrys = src[t][0]
                src[t].pop(0)
                if not arrys.shape == v.shape:
                    arrys = arrys.T
                    print('\033[1;35m {} shape not equal. \033[0m'.format(k))
                    # continue
                ones = torch.Tensor(arrys)
                dst_state_dict[k] = torch.nn.Parameter(ones)
                flag = True
                print('-[sucess] {} to {}'.format(name, k))
                break
        assert flag, '\033[1;35m {} no match. \033[0m'.format(k)

    torch.save(dst_state_dict, 'output/resnet18/tyre/output.pth')