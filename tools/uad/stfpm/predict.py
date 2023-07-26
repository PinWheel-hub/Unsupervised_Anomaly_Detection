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
import datetime
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)
from qinspector.uad.models.stfpm_torch import ResNet_MS3_torch
from qinspector.cvlib.uad_configs import ConfigParser
from qinspector.uad.utils.utils_torch import plot_fig


def argsparser():
    parser = argparse.ArgumentParser("STFPM")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path of config",
        required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument(
        "--img_path", type=str, default=None, help="picture path")
    parser.add_argument('--resize', type=list, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument(
        "--model_path", type=str, default=None, help="student model_path")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save_pic", type=bool, default=None)

    return parser.parse_args()


def main():
    args = argsparser()
    config_parser = ConfigParser(args)
    args = config_parser.parser()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)

    # build model
    teacher = ResNet_MS3_torch(arch=args.backbone, pretrained=True).cuda()
    student = ResNet_MS3_torch(arch=args.backbone, pretrained=False).cuda()

    # build datasets
    transform = transforms.Compose([
        transforms.Resize(args.resize), transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    saved_dict = torch.load(args.model_path)
    print('load ' + args.model_path)
    student.load_state_dict(saved_dict)
    if(os.path.isdir(args.img_path)):
        img_files = os.listdir(args.img_path)
        results = []
        for img_file in img_files:
            img = Image.open(os.path.join(args.img_path, img_file)).convert('RGB')
            img = transform(img).unsqueeze(0)
            teacher.eval()
            student.eval()
            with torch.no_grad():
                t_feat = teacher(img.cuda())
                s_feat = student(img.cuda())
            score_map = 100.
            for j in range(len(t_feat)):
                t_feat[j] = F.normalize(t_feat[j], dim=1)
                s_feat[j] = F.normalize(s_feat[j], dim=1)
                sm = torch.sum((t_feat[j] - s_feat[j])**2, 1, keepdim=True)
                sm = F.interpolate(
                    sm,
                    size=(args.resize[0], args.resize[1]),
                    mode='bilinear',
                    align_corners=False)
                # aggregate score map by element-wise product
                score_map = score_map * sm  # layer map
            print(img_file, score_map.cpu().numpy().max())
            if args.save_pic:
                save_name = os.path.join(args.result_path, f'{os.path.splitext(img_file)[0]}: {score_map.max()}.png')
                plot_fig(img.numpy(),
                        score_map.squeeze(1).cpu(), None, args.threshold, save_name,
                        args.category, args.save_pic)
            results.append(score_map.cpu().numpy().max())
        print(results)
    else:
        img = Image.open(args.img_path).convert('RGB')
        img = transform(img).unsqueeze(0)
        teacher.eval()
        student.eval()
        with torch.no_grad():
            t_feat = teacher(img.cuda())
            s_feat = student(img.cuda())
        score_map = 1.
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j])**2, 1, keepdim=True)
            sm = F.interpolate(
                sm,
                size=(args.resize[0], args.resize[1]),
                mode='bilinear',
                align_corners=False)
            # aggregate score map by element-wise product
            score_map = score_map * sm  # layer map
        print(score_map.max(), score_map.min())
        if args.save_pic:
            save_name = os.path.join(args.result_path, f'{os.path.splitext(os.path.basename(args.img_path))[0]}: {score_map.max()}.png')
            plot_fig(img.numpy(),
                    score_map.squeeze(1).cpu(), None, args.threshold, save_name,
                    args.category, args.save_pic)

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '\t' +
            "Predict :  Picture {}".format(args.img_path) + " done!")


if __name__ == "__main__":
    main()
