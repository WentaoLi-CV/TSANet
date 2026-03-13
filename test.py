import os
import warnings

import numpy as np
from PIL import Image
import cv2
import clip
import torch
from torchvision.transforms import functional as F
from model.TSANet import MultiModel as create_model

import argparse
from tqdm import tqdm
from utils.calculate_metric_gpu import metric_gpu
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    root_path = args.dataset_path
    save_path = args.save_path
    save_path_image = os.path.join(save_path, "image")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    if os.path.exists(save_path_image) is False:
        os.makedirs(save_path_image)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vis_root = os.path.join(root_path, args.vis_dir)
    ir_root = os.path.join(root_path, args.ir_dir)
    vis_paths = list_image_files(vis_root)
    ir_paths = list_image_files(ir_root)
    have_text = True
    text_vis_root = os.path.join(root_path, args.text_vis_dir)
    text_ir_root = os.path.join(root_path, args.text_ir_dir)
    if os.path.exists(text_vis_root) is False or os.path.exists(text_ir_root) is False:
        have_text = False
    else:
        text_vis_paths = list_text_files(text_vis_root)
        text_ir_paths = list_text_files(text_ir_root)

    print("Find the number of visible image: {},  the number of the infrared image: {}".format(len(vis_paths), len(ir_paths)))
    assert len(vis_paths) == len(ir_paths), "The number of the source images does not match!"

    print("Begin to run!")
    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device=device)
        model = create_model(model_clip).to(device)

        model_weight_path = args.weights_path
        model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
        model.eval()

    for i in tqdm(range(len(vis_paths))):
        ir_path = ir_paths[i]
        vi_path = vis_paths[i]
        img_name = vi_path.split("/")[-1]
        assert os.path.exists(ir_path), "file: '{}' dose not exist.".format(ir_path)
        assert os.path.exists(vi_path), "file: '{}' dose not exist.".format(vi_path)
        ir = Image.open(ir_path).convert(mode="RGB")
        vi = Image.open(vi_path).convert(mode="RGB")

        if have_text:
            text_vis_path = text_vis_paths[i]
            text_ir_path = text_ir_paths[i]
            with open(text_vis_path, 'r', encoding='utf-8') as f:
                text_vis = f.readline().strip()
            with open(text_ir_path, 'r', encoding='utf-8') as f:
                text_ir = f.readline().strip()
        else:
            text_vis = f'There is a {args.vis_dir} image.'
            text_ir = f'There is a {args.ir_dir} image.'

        width, height = vi.size
        new_width = max(16, (width // 16) * 16)
        new_height = max(16, (height // 16) * 16)
        ir = ir.resize((new_width, new_height))
        vi = vi.resize((new_width, new_height))

        ir = F.to_tensor(ir)
        vi = F.to_tensor(vi)

        ir = ir.unsqueeze(0).to(device)
        vi = vi.unsqueeze(0).to(device)
        with torch.no_grad():
            text_vis = clip.tokenize(text_vis).to(device)
            text_ir = clip.tokenize(text_ir).to(device)
            loss, fi = model(vi, ir, text_vis, text_ir)
            # # TODO match the original size
            fi = torch.nn.functional.interpolate(
                fi,
                size=(height, width),
                mode='bicubic',
                align_corners=False,
                antialias=False
            )
            fused_img_Y = tensor2numpy(fi)
            save_pic(fused_img_Y, save_path_image, img_name)

    print("Finish! The results are saved in {}.".format(save_path))

    # Metric
    print(f'Calculated by GPU')
    metric_gpu(root_path, save_path, args.vis_dir, args.ir_dir, device=device)


def tensor2numpy(img_tensor):
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    return img


def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    base_name = os.path.splitext(index)[0]
    save_path = os.path.join(path, f"{base_name}.png")
    cv2.imwrite(save_path, outputpic)


def list_image_files(root_dir):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [
        os.path.join(root_dir, i)
        for i in sorted(os.listdir(root_dir))
        if os.path.splitext(i)[-1].lower() in image_exts
    ]


def list_text_files(root_dir):
    return [
        os.path.join(root_dir, i)
        for i in sorted(os.listdir(root_dir))
        if os.path.splitext(i)[-1].lower() == ".txt"
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/public/lwt/PycharmProjects/Text-IF-main/dataset/ACCV/eval', help='test data root path')
    parser.add_argument('--weights_path', type=str,
                        default='experiments/accv_multitext_multiscale_imgfus1_textsam_max01/weights/checkpoint.pth',
                        help='initial weights path')
    parser.add_argument('--save_path', type=str, default='./results/test', help='output save image path')
    parser.add_argument('--vis_dir', type=str, default='Visible', help='visible image folder name')
    parser.add_argument('--ir_dir', type=str, default='Infrared', help='infrared image folder name')
    parser.add_argument('--text_vis_dir', type=str, default='text_vis', help='visible text folder name')
    parser.add_argument('--text_ir_dir', type=str, default='text_ir', help='infrared text folder name')

    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    opt = parser.parse_args()
    main(opt)
