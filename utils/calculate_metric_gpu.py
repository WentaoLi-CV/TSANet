import os
import math
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def append_to_file(path, data):
    with open(path, 'a') as f:
        f.write(data + '\n')


def _to_gray_tensor(path, device):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    bgr = bgr.astype('float32')
    gray = np.round(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    tensor = torch.from_numpy(gray).to(device)
    return tensor


def _entropy(img):
    img = img.round().clamp(0, 255)
    hist = torch.histc(img, bins=256, min=0, max=255)
    p = hist / hist.sum()
    p = p[p > 0]
    return -(p * torch.log2(p)).sum()


def _std(img):
    return img.std(unbiased=False)


def _sf(img):
    diff_h = img[:, 1:] - img[:, :-1]
    diff_v = img[1:, :] - img[:-1, :]
    return torch.sqrt(diff_h.pow(2).mean() + diff_v.pow(2).mean())


def _mutual_info(image_f, image_a, image_b):
    def _mi_pair(x, y, bins=256):
        x = x.round().clamp(0, bins - 1).to(torch.long)
        y = y.round().clamp(0, bins - 1).to(torch.long)
        flat_x = x.view(-1)
        flat_y = y.view(-1)
        joint_index = flat_x * bins + flat_y
        joint_hist = torch.bincount(joint_index, minlength=bins * bins).float()
        joint_hist = joint_hist.view(bins, bins)
        pxy = joint_hist / joint_hist.sum()
        px = pxy.sum(dim=1, keepdim=True)
        py = pxy.sum(dim=0, keepdim=True)
        denom = px * py
        mask = pxy > 0
        return (pxy[mask] * torch.log2(pxy[mask] / denom[mask])).sum()

    return _mi_pair(image_f, image_a) + _mi_pair(image_f, image_b)


def _scd(image_f, image_a, image_b):
    eps = 1e-10
    imgf_a = image_f - image_a
    imgf_b = image_f - image_b
    r_af = ((image_a - image_a.mean()) * (imgf_b - imgf_b.mean())).sum() / (
            torch.sqrt(((image_a - image_a.mean()) ** 2).sum() * ((imgf_b - imgf_b.mean()) ** 2).sum()) + eps
    )
    r_bf = ((image_b - image_b.mean()) * (imgf_a - imgf_a.mean())).sum() / (
            torch.sqrt(((image_b - image_b.mean()) ** 2).sum() * ((imgf_a - imgf_a.mean()) ** 2).sum()) + eps
    )
    return r_af + r_bf


def _gaussian_kernel(size, sigma, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


def _viff_pair(ref, dist):
    sigma_nsq = 2
    eps = 1e-10
    num = 0.0
    den = 0.0

    ref = ref.unsqueeze(0).unsqueeze(0)
    dist = dist.unsqueeze(0).unsqueeze(0)

    for scale in range(1, 5):
        size = 2 ** (4 - scale + 1) + 1
        sigma = size / 5.0
        win = _gaussian_kernel(size, sigma, ref.device, ref.dtype).unsqueeze(0).unsqueeze(0)

        if scale > 1:
            ref = F.conv2d(ref, win)
            dist = F.conv2d(dist, win)
            ref = ref[:, :, ::2, ::2]
            dist = dist[:, :, ::2, ::2]

        mu1 = F.conv2d(ref, win)
        mu2 = F.conv2d(dist, win)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(ref * ref, win) - mu1_sq
        sigma2_sq = F.conv2d(dist * dist, win) - mu2_sq
        sigma12 = F.conv2d(ref * dist, win) - mu1_mu2

        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g = torch.where(sigma1_sq < eps, torch.zeros_like(g), g)
        sv_sq = torch.where(sigma1_sq < eps, sigma2_sq, sv_sq)
        sigma1_sq = torch.where(sigma1_sq < eps, torch.zeros_like(sigma1_sq), sigma1_sq)

        g = torch.where(sigma2_sq < eps, torch.zeros_like(g), g)
        sv_sq = torch.where(sigma2_sq < eps, torch.zeros_like(sv_sq), sv_sq)

        sv_sq = torch.where(g < 0, sigma2_sq, sv_sq)
        g = torch.clamp(g, min=0)
        sv_sq = torch.clamp(sv_sq, min=eps)

        num += torch.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)).sum()
        den += torch.log10(1 + sigma1_sq / sigma_nsq).sum()

    vifp = num / den
    if torch.isnan(vifp):
        return torch.tensor(1.0, device=ref.device)
    return vifp


def _viff(image_f, image_a, image_b):
    return _viff_pair(image_a, image_f) + _viff_pair(image_b, image_f)


def _ssim_pair(image_x, image_y, window_size=11, sigma=1.5, k1=0.01, k2=0.03):
    eps = 1e-10
    data_range = torch.clamp(image_x.max() - image_x.min(), min=eps)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    win = _gaussian_kernel(window_size, sigma, image_x.device, image_x.dtype).unsqueeze(0).unsqueeze(0)
    x = image_x.unsqueeze(0).unsqueeze(0)
    y = image_y.unsqueeze(0).unsqueeze(0)

    mu_x = F.conv2d(x, win, padding=window_size // 2)
    mu_y = F.conv2d(y, win, padding=window_size // 2)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, win, padding=window_size // 2) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, win, padding=window_size // 2) - mu_y_sq
    sigma_xy = F.conv2d(x * y, win, padding=window_size // 2) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return (numerator / denominator).mean()


def _ssim(image_f, image_a, image_b):
    return (_ssim_pair(image_f, image_a) + _ssim_pair(image_f, image_b)) / 2


def _qabf_get_array(img):
    h1 = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device)
    h2 = torch.tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=img.dtype, device=img.device)
    h3 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device)

    img_4d = img.unsqueeze(0).unsqueeze(0)
    SAx = F.conv2d(img_4d, h3.unsqueeze(0).unsqueeze(0), padding=1).squeeze(0).squeeze(0)
    SAy = F.conv2d(img_4d, h1.unsqueeze(0).unsqueeze(0), padding=1).squeeze(0).squeeze(0)

    gA = torch.sqrt(SAx * SAx + SAy * SAy)
    aA = torch.zeros_like(img)
    aA = torch.where(SAx == 0, torch.tensor(math.pi / 2, device=img.device, dtype=img.dtype), aA)
    aA = torch.where(SAx != 0, torch.atan(SAy / SAx), aA)
    return gA, aA


def _qabf_get_qabf(aA, gA, aF, gF):
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    GAF = torch.zeros_like(aA)
    GAF = torch.where(gA > gF, gF / gA, GAF)
    GAF = torch.where(gA == gF, gF, GAF)
    GAF = torch.where(gA < gF, gA / gF, GAF)

    AAF = 1 - torch.abs(aA - aF) / (math.pi / 2)
    QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))
    return QgAF * QaAF


def _qabf(image_f, image_a, image_b):
    eps = 1e-10
    gA, aA = _qabf_get_array(image_a)
    gB, aB = _qabf_get_array(image_b)
    gF, aF = _qabf_get_array(image_f)
    qAF = _qabf_get_qabf(aA, gA, aF, gF)
    qBF = _qabf_get_qabf(aB, gB, aF, gF)
    deno = (gA + gB).sum()
    nume = (qAF * gA + qBF * gB).sum()
    return nume / (deno + eps)

def process_image_gpu(img_name, dataset_path, save_path_image, vis_dir, ir_dir, device):
    ir_path = os.path.join(dataset_path, ir_dir, img_name)
    vi_path = os.path.join(dataset_path, vis_dir, img_name)
    base_name = os.path.splitext(img_name)[0]
    fi_path = os.path.join(save_path_image, f"{base_name}.png")

    vi = _to_gray_tensor(vi_path, device)
    ir = _to_gray_tensor(ir_path, device)
    fi = _to_gray_tensor(fi_path, device)

    metrics = torch.stack([
        _entropy(fi),
        _std(fi),
        _sf(fi),
        _mutual_info(fi, ir, vi),
        _scd(fi, ir, vi),
        _viff(fi, ir, vi),
        _qabf(fi, ir, vi),
        _ssim(fi, ir, vi),
    ])
    return img_name, metrics


def metric_gpu(dataset_path, save_path, vis_dir, ir_dir, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path_image = os.path.join(save_path, "image")
    ir_dir = os.path.join(dataset_path, "Infrared")
    vi_dir = os.path.join(dataset_path, "Visible")
    for required_dir in (ir_dir, vi_dir, save_path_image):
        if not os.path.isdir(required_dir):
            raise FileNotFoundError(f"Required directory does not exist: {required_dir}")

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    img_list = [f for f in os.listdir(ir_dir) if f.lower().endswith(image_exts)]

    results = []
    for img_name in tqdm(img_list, desc="Processing Images (GPU)"):
        img_name, metrics = process_image_gpu(img_name, dataset_path, save_path_image, vis_dir, ir_dir, device)
        results.append((img_name, metrics.detach().cpu().numpy()))

    df = pd.DataFrame(
        [[img_name] + list(metrics) for img_name, metrics in results],
        columns=["Image", "EN", "SD", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM"],
    )

    csv_path = os.path.join(save_path, "image_metrics.csv")
    df.to_csv(csv_path, index=False)

    metric_result = df.iloc[:, 1:].mean().values

    print("\t\t EN\t SD\t SF\t MI\t SCD\t VIF\t Qabf\t SSIM")
    print('Metric_GPU' + '\t'
          + str(np.round(metric_result[0], 2)) + '\t'
          + str(np.round(metric_result[1], 2)) + '\t'
          + str(np.round(metric_result[2], 2)) + '\t'
          + str(np.round(metric_result[3], 2)) + '\t'
          + str(np.round(metric_result[4], 2)) + '\t'
          + str(np.round(metric_result[5], 2)) + '\t'
          + str(np.round(metric_result[6], 2)) + '\t'
          + str(np.round(metric_result[7], 2)) + '\t'
          )
    print("=" * 80)

    log_file = f'{save_path}/test_metric'
    head = "The test result of " + save_path + ' :'
    header = "\t\t EN\t SD\t SF\t MI\t SCD\t VIF\t Qabf\t SSIM"
    data_line = (f"Metric_GPU\t"
                 + str(np.round(metric_result[0], 3)) + '\t'
                 + str(np.round(metric_result[1], 3)) + '\t'
                 + str(np.round(metric_result[2], 3)) + '\t'
                 + str(np.round(metric_result[3], 3)) + '\t'
                 + str(np.round(metric_result[4], 3)) + '\t'
                 + str(np.round(metric_result[5], 3)) + '\t'
                 + str(np.round(metric_result[6], 3)) + '\t'
                 + str(np.round(metric_result[7], 3)) + '\t'
                 )
    start = "=" * 80

    append_to_file(log_file, start)
    append_to_file(log_file, head)
    append_to_file(log_file, header)
    append_to_file(log_file, data_line)
    append_to_file(log_file, start)


def main(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metric_gpu(args.dataset_path, args.save_path, args.vis_dir, args.ir_dir, device=device)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="dataset/ACCV/eval", help='test data root path')
    parser.add_argument('--save_path', type=str, default='./results/test', help='output save image path')
    parser.add_argument('--vis_dir', type=str, default='t1', help='visible image folder name')
    parser.add_argument('--ir_dir', type=str, default='t2', help='infrared image folder name')
    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    args = parser.parse_args()
    main(args)
