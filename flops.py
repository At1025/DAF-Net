from models.yolo_test import Model
import argparse
import torch
from utils.torch_utils import select_device
from utils.general import check_file, set_logging

from thop.profile import profile
from thop import clever_format
from thop.vision.basic_hooks import zero_ops

import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


# ==============================
# 🚨 关键：注册无法解析模块
# ==============================
def register_custom_ops():
    """
    将动态算子替换为 zero_ops，
    保证 THOP 不会在 DFConv / FDFTM 中崩溃
    """
    return {
        "DFConv": zero_ops,
        "FDFTM": zero_ops,
        "LFEM": zero_ops,
        "CMDFB": zero_ops,
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg',
        type=str,
        default='/data/MOD/models/transformer/CDAM+FCFM.yaml',
        help='model.yaml'
    )
    parser.add_argument(
        '--device',
        default='1',
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
    )

    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)

    set_logging()
    device = select_device(opt.device)
    print(device)

    # ==============================
    # model
    # ==============================
    model = Model(opt.cfg).to(device)
    model.eval()   # ⭐必须加：避免BN/Dropout影响profile

    # ==============================
    # dummy inputs (RGB + IR)
    # ==============================
    input_rgb = torch.randn(1, 3, 640, 640).to(device)
    input_ir  = torch.randn(1, 3, 640, 640).to(device)

    # ==============================
    # FLOPs profile（安全版本）
    # ==============================
    custom_ops = register_custom_ops()

    macs, params = profile(
        model,
        inputs=(input_rgb, input_ir),
        custom_ops=custom_ops,
        verbose=False
    )

    # ==============================
    # format output
    # ==============================
    macs, params = clever_format([2 * macs, params], "%.4f")

    print("\n==============================")
    print(" CAMF FLOPs PROFILE RESULT")
    print("==============================")
    print("MACs:   ", macs)
    print("Params: ", params)