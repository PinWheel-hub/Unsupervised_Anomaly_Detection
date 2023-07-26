import subprocess
import os
import sys
import argparse

parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)
from qinspector.uad.utils.utils_torch import str2bool
from qinspector.cvlib.uad_configs import ConfigParser

def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "option",
        type=str,
        default=None)
    parser.add_argument(
        "config",
        type=str,
        default=None,
        help="Path of config")
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

if __name__ == '__main__':
    args = argsparser()
    config_parser = ConfigParser(args, if_print=False)
    args = config_parser.parser()
    script_name = os.path.join('tools', 'uad', f"{args.model}", f'{args.option}.py')
    script_args = ["--config", args.config]
    subprocess.run(["python", script_name] + script_args)