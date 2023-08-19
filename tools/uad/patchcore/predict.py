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

import os
import sys
import random
import argparse
import datetime

import numpy as np
from PIL import Image

import torch

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)
from qinspector.uad.datasets import mvtec_torch
from qinspector.uad.models.patchcore_torch import get_model
from qinspector.uad.utils.utils_torch import plot_fig, str2bool
from qinspector.cvlib.uad_configs import ConfigParser


def argsparser():
    parser = argparse.ArgumentParser('PatchCore')
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path of config",
        required=True)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help="specify model path if needed")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="category name for MvTec AD dataset")
    parser.add_argument('--resize', type=list or tuple, default=None)
    parser.add_argument('--crop_size', type=list or tuple, default=None)
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="backbone model arch, one of [resnet18, resnet50, wide_resnet50_2]")
    parser.add_argument("--k", type=int, default=None, help="feature used")
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="projection method, one of [sample,ortho]")
    parser.add_argument("--save_pic", type=str2bool, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)

    parser.add_argument("--norm", type=str2bool, default=True)
    return parser.parse_args()


def main():
    args = argsparser()
    config_parser = ConfigParser(args)
    args = config_parser.parser()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)

    class_name = args.category
    # assert class_name in mvtec.CLASS_NAMES
    print("Testing model for {}".format(class_name))
    # build model
    model = get_model(args.method)(arch=args.backbone,
                                   pretrained=False,
                                   k=args.k,
                                   method=args.method).cuda()
    model.eval()
    state = torch.load(args.model_path)
    model.model.load_state_dict(state["params"])
    model.load(state["stats"])
    model.eval()

    # build data
    MVTecDataset = mvtec_torch.MVTecDataset_torch(is_predict=True, resize=args.resize, cropsize=args.crop_size)
    transform_x = MVTecDataset.get_transform_x()
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Starting eval model...")
    if(os.path.isdir(args.img_path)):
        img_files = os.listdir(args.img_path)
        results = []
        for img_file in img_files:
            x = Image.open(os.path.join(args.img_path, img_file)).convert('RGB')
            x = transform_x(x).unsqueeze(0)
            score_map = predict(args, model, x, img_file)
            results.append(score_map.max())
        print(results)
    else:
        x = Image.open(args.img_path).convert('RGB')
        x = transform_x(x).unsqueeze(0)
        score_map = predict(args, model, x, os.path.basename(args.img_path))
        print(score_map.max(), score_map.min())
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
          "Predict :  Picture {}".format(args.img_path) + " done!")


def predict(args, model, x, img_name):
    # extract test set features
    # model prediction
    start_time = datetime.datetime.now()
    out = model(x.cuda())
    out = model.project(out)
    score_map, image_score = model.generate_scores_map(out, x.shape[-2:])
    end_time = datetime.datetime.now()
    print("代码执行时间：", (end_time - start_time).total_seconds(), "秒")
    # score_map = np.concatenate(score_map, 0)

    # Normalization
    if args.norm:
        max_score = score_map.max()
        min_score = score_map.min()
        # (score_map - min_score) / (max_score - min_score)
    print(os.path.basename(img_name), score_map.max())
    if args.save_pic:
        save_path = os.path.join(args.result_path, f'{os.path.splitext(img_name)[0]}: {score_map.max()}.png')
        plot_fig(x.numpy(), score_map, None, args.threshold, save_path,
             args.category, args.save_pic)
    return score_map


if __name__ == '__main__':
    main()
