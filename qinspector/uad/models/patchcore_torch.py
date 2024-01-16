# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, wide_resnet50_2, resnet101

from scipy.ndimage import gaussian_filter
from qinspector.cvlib.workspace import register
from qinspector.uad.utils.utils_torch import cdist, my_cdist, cholesky_inverse, mahalanobis, mahalanobis_einsum, orthogonal, svd_orthogonal
from qinspector.uad.utils.k_center_greedy_torch import KCenterGreedy, my_KCenterGreedy

from typing import Tuple
class GAN(nn.Module):
    """Encoder Network.

    Args:
        input_size (tuple[int, int]): Size of input image
        latent_vec_size (int): Size of latent vector z
        num_input_channels (int): Number of input channels in the image
        n_features (int): Number of features per convolution layer
        extra_layers (int): Number of extra layers since the network uses only a single encoder layer by default.
            Defaults to 0.
    """

    def __init__(
        self,
        input_size: Tuple[int, int]=(256, 256),
        latent_vec_size: int=100,
        num_input_channels: int=3,
        n_features: int=64,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()

        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{num_input_channels}-{n_features}",
            nn.Conv2d(num_input_channels, n_features, kernel_size=4, stride=2, padding=4, bias=False),
        )
        self.input_layers.add_module(f"initial-relu-{n_features}", nn.LeakyReLU(0.2, inplace=True))

        # Extra Layers
        self.extra_layers = nn.Sequential()

        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_features}-conv",
                nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-batchnorm", nn.BatchNorm2d(n_features))
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-relu", nn.LeakyReLU(0.2, inplace=True))

        # Create pyramid features to reach latent vector
        self.pyramid_features = nn.Sequential()
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_features
            out_features = n_features * 2
            self.pyramid_features.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.pyramid_features.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.pyramid_features.add_module(f"pyramid-{out_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            n_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Final conv
        if add_final_conv_layer:
            self.final_conv_layer = nn.Conv2d(
                n_features,
                latent_vec_size,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )

    def forward(self, input_tensor):
        """Return latent vectors."""

        output = self.input_layers(input_tensor)
        output = self.extra_layers(output)
        output = self.pyramid_features(output)
        if self.final_conv_layer is not None:
            output = self.final_conv_layer(output)

        return output

gan =  GAN()

models = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "wide_resnet50_2": wide_resnet50_2,
    "resnet101": resnet101,
    "gan": gan
}
fins = {
    "resnet18": 448,
    "resnet50": 1792,
    "wide_resnet50_2": 1792,
    "resnet101": 1792,
    "gan": 64
}


def get_projection(fin, fout, method='ortho'):
    if 'sample' == method:
        W = torch.randperm(fin)[:fout]
        # W = paddle.eye(fin)[W.tolist()].T
    elif 'coreset' == method or 'local_coreset' == method:
        W = None
    elif 'h_sample' == method:
        s = torch.randperm(fin // 7)[:fout // 3].tolist() \
            + (fin // 7 + torch.randperm(fin // 7 * 2)[:fout // 3]).tolist() \
            + (fin // 7 * 3 + torch.randperm(fin // 7 * 4)[:(fout - fout // 3 * 2)]).tolist()
        W = torch.eye(fin)[s].T
    elif 'ortho' == method:
        W = orthogonal(fin, fout)
    elif 'svd_ortho' == method:
        W = svd_orthogonal(fin, fout)
    elif 'gaussian' == method:
        W = torch.randn(fin, fout)
    return W


class PaDiMPlus_torch(nn.Module):
    def __init__(self, arch='resnet18', pretrained=True, k=10, method='sample'):
        super().__init__()
        if isinstance(arch, type(None)) or isinstance(pretrained, type(None)):
            self.model = None
            print('Inference mode')
        else:
            assert arch in models.keys(), 'arch {} not supported'.format(arch)
            if arch == 'gan':
                self.model = models[arch]
            else:
                self.model = models[arch](pretrained)
                del self.model.layer4, self.model.fc, self.model.avgpool
            self.model.eval()
            print(
                f'model {arch}, nParams {sum([w.numel() for w in self.model.parameters()])}'
            )
            self.arch = arch
            self.method = method
            self.fin = fins[arch]
            self.k = k
            self.projection = None
            self.reset_stats()

    def init_projection(self):
        self.projection = get_projection(fins[self.arch], self.k, self.method)

    def load(self, state):
        self.mean = state['mean']
        self.inv_covariance = state['inv_covariance']
        self.projection = state['projection']

    def reset_stats(self, set_None=True):
        if set_None:
            self.mean = None
            self.inv_covariance = None
        else:
            self.mean = torch.zeros_like(self.mean)
            self.inv_covariance = torch.zeros_like(self.inv_covariance)

    def set_dist_params(self, mean, inv_cov):
        self.mean, self.inv_covariance = mean, inv_cov

    @torch.no_grad()
    def project_einsum(self, x):
        return torch.einsum('bchw, cd -> bdhw', x, self.projection)
        # if self.method == 'ortho':
        #    return paddle.einsum('bchw, cd -> bdhw', x, self.projection)
        # else: #self.method == 'PaDiM':
        #    return paddle.index_select(embedding,  self.projection, 1)

    @torch.no_grad()
    def project(self, x, return_HWBC=False):
        if isinstance(self.projection, type(None)):
            return x.permute((2, 3, 0, 1)) if return_HWBC else x
        B, C, H, W = x.shape
        if len(self.projection.shape) == 1:
            x = torch.index_select(x, 1, self.projection)
            if return_HWBC: x = x.permute((2, 3, 0, 1))
            return x
        else:
            if return_HWBC:
                x = x.permute((2, 3, 0, 1))
                return x @self.projection
                result = []  # paddle.zeros((B, self.k, H, W))
                for i in range(H):
                    # result[i] = paddle.einsum('chw, cd -> dhw', x[i], self.projection)
                    # result[i,:,:,:] = x[i] @self.projection.T
                    result.append(x[i] @self.projection.T)
                result = paddle.stack(result)
                return result
            result = []  # paddle.zeros((B, self.k, H, W))
            x = x.reshape((B, C, H * W))
            for i in range(B):
                # result[i] = paddle.einsum('chw, cd -> dhw', x[i], self.projection)
                # result[i] = (self.projection.T @ x[i]).reshape((self.k, H, W))
                result.append((self.projection.T @x[i]).reshape((self.k, H, W)))
            result = torch.stack(result)
            return result

    @torch.no_grad()
    def forward_res(self, x):
        res = []
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            res.append(x)
            x = self.model.layer2(x)
            res.append(x)
            x = self.model.layer3(x)
            res.append(x)
        return res

    @torch.no_grad()
    def forward(self, x):
        res = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        res.append(x)
        x = self.model.layer2(x)
        res.append(x)
        x = self.model.layer3(x)
        res.append(x)
        x = res
        for i in range(1, len(x)):
            x[i] = F.interpolate(x[i], scale_factor=2**i, mode="nearest")
        # print([i.shape for i in x])
        x = torch.cat(x, 1)
        # x = self.project(x)
        return x

    @torch.no_grad()
    def forward_score(self, x):
        return self.generate_scores_map(self.get_embedding(x), x.shape)

    @torch.no_grad()
    def compute_stats_einsum(self, outs):
        # calculate multivariate Gaussian distribution
        B, C, H, W = outs.shape
        mean = outs.mean(0)  # mean chw
        outs -= mean
        cov = torch.einsum('bchw, bdhw -> hwcd', outs, outs) / (
            B - 1)  # covariance hwcc
        self.compute_inv(mean, cov)

    @torch.no_grad()
    def compute_stats_(self, embedding):
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding.shape
        mean = torch.mean(embedding, axis=0)
        embedding = embedding.reshape((B, C, H * W))
        cov = np.empty((C, C, H * W))
        for i in tqdm(range(H * W)):
            cov[:, :, i] = np.cov(embedding[:, :, i].numpy(), rowvar=False)
        cov = torch.tensor(cov.reshape(C, C, H, W).permute((2, 3, 0, 1)))
        self.compute_inv(mean, cov)

    @torch.no_grad()
    def compute_stats_np(self, embedding):
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding.shape
        mean = torch.mean(embedding, axis=0)
        embedding = embedding.reshape((B, C, H * W)).numpy()
        inv_covariance = np.empty((H * W, C, C), dtype='float32')
        I = np.identity(C)
        for i in tqdm(range(H * W)):
            inv_covariance[i, :, :] = np.linalg.inv(
                np.cov(embedding[:, :, i], rowvar=False) + 0.01 * I)
        inv_covariance = torch.tensor(inv_covariance.reshape(
            H, W, C, C)).astype('float32')
        self.set_dist_params(mean, inv_covariance)

    @torch.no_grad()
    def compute_stats(self, embedding):
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding.shape
        mean = torch.mean(embedding, axis=0)
        embedding -= mean
        embedding = embedding.permute((2, 3, 0, 1))  # hwbc
        inv_covariance = []  # paddle.zeros((H, W, C, C), dtype='float32')
        I = torch.eye(C)
        for i in tqdm(range(H), desc='compute distribution stats'):
            inv_covariance.append(
                torch.einsum('wbc, wbd -> wcd', embedding[i], embedding[i]) / (
                    B - 1) + 0.01 * I)
            inv_covariance[-1] = cholesky_inverse(
                inv_covariance[-1])  # paddle.inverse(inv_covariance[-1])#
        inv_covariance = torch.stack(inv_covariance).reshape(
            (H, W, C, C)).astype('float32')
        self.set_dist_params(mean, inv_covariance)

    @torch.no_grad()
    def compute_stats_incremental(self, out):
        # calculate multivariate Gaussian distribution
        H, W, B, C = out.shape
        if isinstance(self.inv_covariance, type(None)):
            self.mean = torch.zeros((H, W, C), dtype='float32')
            self.inv_covariance = torch.zeros((H, W, C, C), dtype='float32')

        self.mean += out.sum(2)  # mean hwc
        # cov = paddle.einsum('bchw, bdhw -> hwcd', outs, outs)# covariance hwcc
        for i in range(H):
            self.inv_covariance[i, :, :, :] += torch.einsum(
                'wbc, wbd -> wcd', out[i, :, :, :], out[i, :, :, :])
        # return mean, cov, B

    def compute_inv_incremental(self, B, eps=0.01):
        c = self.mean.shape[0]
        # if self.inv_covariance == None:
        self.mean /= B  # hwc
        self.inv_covariance /= B
        # covariance hwcc  #.permute((2,3, 0, 1)))
        self.inv_covariance -= torch.einsum('hwc, hwd -> hwcd', self.mean,
                                             self.mean)
        # covariance = (covariance - B*paddle.einsum('chw, dhw -> hwcd', mean, mean))/(B-1)
        self.compute_inv(
            self.mean.permute((2, 0, 1)), self.inv_covariance, eps)

    def compute_inv_(self, mean, covariance, eps=0.01):
        c = mean.shape[0]
        # if self.inv_covariance == None:
        # covariance hwcc  #.permute((2,3, 0, 1)))
        # self.inv_covariance = paddle.linalg.inv(covariance)
        self.set_dist_params(mean,
                             cholesky_inverse(covariance + eps * torch.eye(c)))

    def compute_inv(self, mean, covariance, eps=0.01):
        c, H, W = mean.shape
        for i in tqdm(range(H), desc='compute inverse covariance'):
            covariance[i, :, :, :] = cholesky_inverse(covariance[i, :, :, :] +
                                                      eps * torch.eye(c))
        self.set_dist_params(mean, covariance)

    def generate_scores_map(self, embedding, out_shape, gaussian_blur=True):
        # calculate distance matrix
        # B, C, H, W = embedding.shape
        # embedding = embedding.reshape((B, C, H * W))

        # calculate mahalanobis distances
        distances = mahalanobis_einsum(embedding, self.mean,
                                       self.inv_covariance)
        score_map = postporcess_score_map(distances, out_shape, gaussian_blur)
        img_score = score_map.reshape(score_map.shape[0], -1).max(axis=1)
        return score_map, img_score

@register
class PatchCore_torch(PaDiMPlus_torch):
    def load(self, state):
        self.memory_bank = state['memory_bank'].cuda()

    def clean_stats(self, set_None=True):
        if set_None:
            self.memory_bank = None
        else:
            self.memory_bank = torch.zeros_like(self.memory_bank)

    def set_dist_params(self, memory_bank):
        self.memory_bank = memory_bank

    @torch.no_grad()
    def forward_res(self, x):
        res = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        res.append(F.avg_pool2d(x, 3, 1, 1))
        x = self.model.layer3(x)
        res.append(F.avg_pool2d(x, 3, 1, 1))
        return res

    @torch.no_grad()
    def forward(self, x):
        if self.arch == 'gan':
            pool = nn.AvgPool2d(2, 2, 0, count_include_pad=True)
            x = self.model.input_layers(x)
            x = self.model.extra_layers(x)
            x = pool(x)
        else:
            pool = nn.AvgPool2d(3, 1, 1, count_include_pad=True)
            res = []
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            res.append(pool(x))
            x = self.model.layer3(x)
            res.append(pool(x))
            x = res
            for i in range(1, len(x)):
                x[i] = F.interpolate(x[i], scale_factor=2**i, mode="nearest")
            # print([i.shape for i in x])
            x = torch.cat(x, 1)
            # x = self.project(x)
        return x

    @torch.no_grad()
    def compute_stats(self, embedding):
        C = embedding.shape[1]
        embedding = embedding.permute((0, 2, 3, 1)).reshape((-1, C))
        print("Creating CoreSet Sampler via k-Center Greedy")
        sampler = KCenterGreedy(embedding, sampling_ratio=self.k / 100)
        print("Getting the coreset from the main embedding.")
        coreset = sampler.sample_coreset()
        print(
            f"Assigning the coreset as the memory bank with shape {coreset.shape}."
        )  # 18032,384
        self.memory_bank = coreset

    def compute_stats_einsum(self, outs):
        raise NotImplementedError

    def compute_stats_incremental(self, out):
        raise NotImplementedError

    def compute_inv_incremental(self, B, eps=0.01):
        raise NotImplementedError

    def project(self, x, return_HWBC=False):
        # no per project
        return x  # super().project(x, return_HWBC)

    def generate_scores_map(self, embedding, out_shape, gaussian_blur=True):
        # Nearest Neighbours distances
        B, C, H, W = embedding.shape
        embedding = embedding.permute((0, 2, 3, 1)).reshape((B, H * W, C))
        distances = self.nearest_neighbors(embedding=embedding, n_neighbors=9)
        distances = distances.permute((2, 0, 1))  # n_neighbors, B, HW
        image_score = []
        for i in range(B):
            image_score.append(
                self.compute_image_anomaly_score(distances[:, i, :]))
        distances = distances[0, :, :].reshape((B, H, W))
        score_map = postporcess_score_map(distances, out_shape, gaussian_blur)
        return score_map, np.array(image_score)

    def nearest_neighbors(self, embedding, n_neighbors: int=9):
        """Compare embedding Features with the memory bank to get Nearest Neighbours distance
        """
        B, HW, C = embedding.shape
        n_coreset = self.memory_bank.shape[0]
        distances = []  # paddle.zeros((B, HW, n_coreset))
        for i in range(B):
            distances.append(torch.cdist(embedding[i, :, :], self.memory_bank, p=2.0))  # euclidean norm
        distances = torch.stack(distances, 0)
        distances, _ = distances.topk(k=n_neighbors, axis=-1, largest=False)
        return distances  # B,

    @staticmethod
    def compute_image_anomaly_score(distance):
        """Compute Image-Level Anomaly Score for one nearest_neighbor distance map.
        """
        # distances[n_neighbors, B, HW]
        max_scores = torch.argmax(distance[0, :])
        confidence = distance[:,
                              max_scores]  # paddle.index_select(distances, max_scores, -1)
        weights = 1 - (torch.max(torch.exp(confidence)) /
                       torch.sum(torch.exp(confidence)))
        score = weights * torch.max(distance[0, :])
        return score.item()
    

@register
class local_coreset(PaDiMPlus_torch):
    def __init__(self, arch='resnet18', pretrained=True, k=10, method='sample', blocks=[8, 8], feature_size=[32, 32]):
        super().__init__(arch=arch, pretrained=pretrained, k=k, method=method)
        self.blocks = blocks
        self.feature_size = feature_size
    def load(self, state):
        self.memory_bank = state['memory_bank'].cuda()

    def clean_stats(self, set_None=True):
        if set_None:
            self.memory_bank = None
        else:
            self.memory_bank = torch.zeros_like(self.memory_bank)

    def set_dist_params(self, memory_bank):
        self.memory_bank = memory_bank

    @torch.no_grad()
    def forward_res(self, x):
        res = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        res.append(F.avg_pool2d(x, 3, 1, 1))
        x = self.model.layer3(x)
        res.append(F.avg_pool2d(x, 3, 1, 1))
        return res

    def attention(self, q, k, v):
        self.num_heads = 2

        width = q.shape[3]
        q = torch.flatten(q, 2, 3)
        k = torch.flatten(k, 2, 3)
        v = torch.flatten(v, 2, 3)
        B, C, N = q.shape
        
        q = q.reshape(B, C // self.num_heads, self.num_heads, N).permute(0, 2, 1, 3).reshape(B * self.num_heads, C // self.num_heads, N)
        k = k.reshape(B, C // self.num_heads, self.num_heads, N).permute(0, 2, 1, 3).reshape(B * self.num_heads, C // self.num_heads, N)
        v = v.reshape(B, C // self.num_heads, self.num_heads, N).permute(0, 2, 1, 3).reshape(B * self.num_heads, C // self.num_heads, N)

        q = torch.reshape(q, (q.shape[0], q.shape[1], q.shape[2]))
        k = torch.reshape(k, (k.shape[0], k.shape[2], k.shape[1]))
        v = torch.reshape(v, (v.shape[0], v.shape[1], v.shape[2]))
        out = torch.matmul(q, k)
        out = torch.nn.functional.softmax(out, dim=-1)
        out = torch.matmul(out, v)
        out = out.reshape(B, C, width, width)
        return out

    @torch.no_grad()
    def forward(self, x):
        avep = nn.AvgPool2d(3, 1, 1, count_include_pad=True)
        maxp = nn.MaxPool2d(3, 1, 1)
        res = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)

        q = avep(x)
        k = avep(x)
        v = avep(x)
        out = self.attention(q, k, v)
        out = out + avep(x)
        # out = torch.concat((out, maxp(x)), dim=1)
        res.append(out)

        x = self.model.layer3(x)

        q = avep(x)
        k = avep(x)
        v = avep(x)
        out = self.attention(q, k, v)
        out = out + avep(x)
        # out = torch.concat((out, pool(x)), dim=1)
        res.append(out)
        
        res[1] = F.interpolate(res[1], scale_factor=2, mode="nearest")
        res = torch.cat(res, 1)
        return res

    @torch.no_grad()
    def compute_stats(self, embedding):
        C = embedding.shape[1]
        my_embedding = embedding.permute((2, 3, 0, 1)).reshape((self.blocks[0], 
                                                                self.feature_size[0] // self.blocks[0], 
                                                                self.blocks[1], 
                                                                self.feature_size[1] // self.blocks[1], 
                                                                -1, C))
        my_embedding = my_embedding.permute((0, 2, 1, 3, 4, 5)).reshape((self.blocks[0], 
                                                                         self.blocks[1], 
                                                                         -1, C))
        # embedding = embedding.permute((0, 2, 3, 1)).reshape((-1, C))
        print("Creating CoreSet Sampler via k-Center Greedy")
        sampler = my_KCenterGreedy(my_embedding, sampling_ratio=self.k / 100)
        print("Getting the coreset from the main embedding.")
        coreset = sampler.sample_coreset()
        print(
            f"Assigning the coreset as the memory bank with shape {coreset.shape}."
        )  # 18032,384
        self.memory_bank = coreset

    def compute_stats_einsum(self, outs):
        raise NotImplementedError

    def compute_stats_incremental(self, out):
        raise NotImplementedError

    def compute_inv_incremental(self, B, eps=0.01):
        raise NotImplementedError

    def project(self, x, return_HWBC=False):
        # no per project
        return x  # super().project(x, return_HWBC)

    def generate_scores_map(self, embedding, out_shape, gaussian_blur=True):
        # Nearest Neighbours distances
        B, C, H, W = embedding.shape
        my_embedding = embedding.permute((2, 3, 0, 1)).reshape((self.blocks[0], 
                                                                self.feature_size[0] // self.blocks[0], 
                                                                self.blocks[1], 
                                                                self.feature_size[1] // self.blocks[1], 
                                                                -1, C))
        my_embedding = my_embedding.permute((0, 2, 1, 3, 4, 5)).reshape((self.blocks[0], 
                                                                         self.blocks[1], 
                                                                         -1, C)).unsqueeze(0)
        # embedding = embedding.permute((0, 2, 3, 1)).reshape((B, H * W, C))
        distances = self.nearest_neighbors(embedding=my_embedding, n_neighbors=9)
        distances = distances.permute((2, 0, 1))  # n_neighbors, B, HW
        image_score = []
        for i in range(B):
            image_score.append(
                self.compute_image_anomaly_score(distances[:, i, :]))
        distances = distances[0, :, :].reshape((B, H, W))
        score_map = postporcess_score_map(distances, out_shape, gaussian_blur)
        return score_map, np.array(image_score)

    def nearest_neighbors(self, embedding, n_neighbors: int=9):
        """Compare embedding Features with the memory bank to get Nearest Neighbours distance
        """
        B, H, W, N, C = embedding.shape
        n_coreset = self.memory_bank.shape[-2]
        distances = []  # paddle.zeros((B, HW, n_coreset))
        for i in range(B):
            distances.append(my_cdist(embedding[i, :, :, :, :], self.memory_bank, feature_size=self.feature_size))  # euclidean norm
        distances = torch.stack(distances, 0)
        distances, _ = distances.topk(k=n_neighbors, axis=-1, largest=False)
        return distances  # B,

    @staticmethod
    def compute_image_anomaly_score(distance):
        """Compute Image-Level Anomaly Score for one nearest_neighbor distance map.
        """
        # distances[n_neighbors, B, HW]
        max_scores = torch.argmax(distance[0, :])
        confidence = distance[:, max_scores]  # paddle.index_select(distances, max_scores, -1)
        weights = 1 - (torch.max(torch.exp(confidence)) /
                       torch.sum(torch.exp(confidence)))
        score = weights * torch.max(distance[0, :])
        # score = torch.sum(confidence) / confidence.shape[0]
        return score.item()


def postporcess_score_map(distances,
                          out_shape,
                          gaussian_blur=True,
                          mode='bilinear'):
    score_map = F.interpolate(
        distances.unsqueeze_(1), size=out_shape, mode=mode,
        align_corners=False).squeeze_(1).cpu().numpy()
    if gaussian_blur:
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

    return score_map


def get_model(method):
    if 'coreset' == method:
        return PatchCore_torch
    elif 'local_coreset' == method:
        return local_coreset
    return PaDiMPlus_torch


if __name__ == '__main__':
    model = PaDiMPlus_torch()
    print(model)
