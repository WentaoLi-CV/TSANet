from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import random
from torchvision.transforms import functional as F
import os


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, vis, ir):
        for t in self.transforms:
            vis, ir = t(vis, ir)
        return vis, ir


class Resize_16(object):
    def __init__(self):
        pass

    def __call__(self, vis, ir):
        width, height = vis.size
        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        vis = F.resize(vis, (new_height, new_width))
        ir = F.resize(ir, (new_height, new_width), interpolation=T.InterpolationMode.NEAREST)

        return vis, ir


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, vis, ir):
        if random.random() < self.flip_prob:
            vis = F.hflip(vis)
            ir = F.hflip(ir)

        return vis, ir


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, vis, ir):
        if random.random() < self.flip_prob:
            vis = F.vflip(vis)
            ir = F.vflip(ir)

        return vis, ir


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vis, ir):
        vis = F.resize(vis, self.size)
        ir = F.resize(ir, self.size)
        return vis, ir


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vis, ir):
        vis = pad_if_smaller(vis, self.size)
        ir = pad_if_smaller(ir, self.size)
        crop_params = T.RandomCrop.get_params(vis, (self.size, self.size))
        vis = F.crop(vis, *crop_params)
        ir = F.crop(ir, *crop_params)

        return vis, ir


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vis, ir):
        vis = F.center_crop(vis, self.size)
        ir = F.center_crop(ir, self.size)
        return vis, ir


class ToTensor(object):
    def __call__(self, vis, ir):
        vis = F.to_tensor(vis)
        ir = F.to_tensor(ir)

        return vis, ir


class MultiModalDataset(Dataset):
    def __init__(self, root, transform=None, phase="train"):
        self.root_path = os.path.join(root, phase)
        image_supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  # 图像格式
        text_supported = [".txt"]  # 文本格式
        self.vis_root = os.path.join(self.root_path, "Visible")
        self.ir_root = os.path.join(self.root_path, "Infrared")
        self.text_root = os.path.join(self.root_path, "text")

        self.vis_paths = [os.path.join(self.vis_root, i) for i in sorted(os.listdir(self.vis_root))
                          if os.path.splitext(i)[-1].lower() in image_supported]
        self.ir_paths = [os.path.join(self.ir_root, i) for i in sorted(os.listdir(self.ir_root))
                         if os.path.splitext(i)[-1].lower() in image_supported]
        self.text_paths = [os.path.join(self.text_root, i) for i in sorted(os.listdir(self.text_root))
                           if os.path.splitext(i)[-1].lower() in text_supported]

        self.phase = phase
        self.transform = transform[phase]

    def __len__(self):
        return len(self.vis_paths)

    def __getitem__(self, idx):
        vis_img = self._load_image(self.vis_paths[idx])
        ir_img = self._load_image(self.ir_paths[idx])
        with open(self.text_paths[idx], 'r', encoding='utf-8') as f:
            text = f.readline().strip()

        vis_img, ir_img = self.transform(vis_img, ir_img)
        name = self.vis_paths[idx].split("/")[-1].split(".")[0]

        return vis_img, ir_img, text, name

    def _load_image(self, path):
        from PIL import Image
        return Image.open(path).convert("RGB")


class MultiTextDataset(Dataset):
    def __init__(
            self,
            root,
            transform=None,
            phase="train",
            vis_dir="t1",
            ir_dir="t2",
            text_vis_dir="text_t1",
            text_ir_dir="text_t2",
    ):
        self.root_path = os.path.join(root, phase)
        self.phase = phase

        self.vis_root = os.path.join(self.root_path, vis_dir)
        self.ir_root = os.path.join(self.root_path, ir_dir)
        self.text_vis_root = os.path.join(self.root_path, text_vis_dir)
        self.text_ir_root = os.path.join(self.root_path, text_ir_dir)

        image_supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.vis_paths = [
            os.path.join(self.vis_root, i)
            for i in sorted(os.listdir(self.vis_root))
            if os.path.splitext(i)[-1].lower() in image_supported
        ]
        self.ir_paths = [
            os.path.join(self.ir_root, i)
            for i in sorted(os.listdir(self.ir_root))
            if os.path.splitext(i)[-1].lower() in image_supported
        ]
        self.text_vis_paths = [
            os.path.join(self.text_vis_root, i)
            for i in sorted(os.listdir(self.text_vis_root))
            if os.path.splitext(i)[-1].lower() == ".txt"
        ]
        self.text_ir_paths = [
            os.path.join(self.text_ir_root, i)
            for i in sorted(os.listdir(self.text_ir_root))
            if os.path.splitext(i)[-1].lower() == ".txt"
        ]
        assert len(self.vis_paths) == len(self.ir_paths), (
            f"Image count mismatch: {len(self.vis_paths)} vis vs {len(self.ir_paths)} ir"
        )
        assert len(self.text_vis_paths) == len(self.text_ir_paths), (
            f"Text count mismatch: {len(self.text_vis_paths)} vis vs {len(self.text_ir_paths)} ir"
        )
        for vp, ip, tvp, tip in zip(self.vis_paths, self.ir_paths, self.text_vis_paths, self.text_ir_paths):
            base_names = [os.path.splitext(os.path.basename(p))[0] for p in [vp, ip, tvp, tip]]
            assert len(set(base_names)) == 1, f"Text name is not same: {base_names} \n: \n{vp}\n{ip}\n{tvp}\n{tip}"

        if isinstance(transform, dict):
            self.transform = transform.get(phase, None)
        else:
            self.transform = transform

    def __getitem__(self, idx):
        vis_img = self._load_image(self.vis_paths[idx])
        ir_img = self._load_image(self.ir_paths[idx])
        with open(self.text_vis_paths[idx], 'r', encoding='utf-8') as f:
            text_vis = f.readline().strip()
        with open(self.text_ir_paths[idx], 'r', encoding='utf-8') as f:
            text_ir = f.readline().strip()

        vis_img, ir_img = self.transform(vis_img, ir_img)
        name = self.vis_paths[idx].split("/")[-1].split(".")[0]

        return vis_img, ir_img, text_vis, text_ir, name

    def _load_image(self, path):
        from PIL import Image
        return Image.open(path).convert("RGB")

    def __len__(self):
        return len(self.vis_paths)