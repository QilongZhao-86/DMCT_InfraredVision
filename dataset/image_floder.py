import glob
import os
import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset


ALIASES = {
    'IRSTD-1K': 'IRSTD-1k',
    'IRSTD-1k': 'IRSTD-1k',
    'NUDT-SIRST': 'NUDT-SIRST',
    'IRSTDID-SKY': 'IRSTDID-SKY',
}

NORMALIZATION = {
    'IRSTD-1k': (
        np.array([0.28450727, 0.28450724, 0.28450724], dtype=np.float32),
        np.array([0.22880708, 0.22880709, 0.22880709], dtype=np.float32)),
    'IRSTDID-SKY': (
        np.array([0.485, 0.456, 0.406], dtype=np.float32),
        np.array([0.229, 0.224, 0.225], dtype=np.float32)),
}


def _resolve_file(folder, item):
    stem = os.path.splitext(item)[0]
    for name in [item] + [stem + ext for ext in ('.png', '.jpg', '.bmp', '.tif', '.tiff')]:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    matches = []
    for ext in ('png', 'jpg', 'bmp', 'tif', 'tiff'):
        matches.extend(glob.glob(os.path.join(folder, stem + '*.' + ext)))
    return sorted(matches)[0] if matches else None


class ImageFolder(Dataset):
    """Official-style preprocessing for the three DMCT IRSTD datasets."""

    def __init__(self, path, data_set='NUDT-SIRST', istraining=True,
                 base_size=256, crop_size=256, suffix='.png', split_file=None):
        if data_set not in ALIASES:
            raise ValueError(f'Unsupported dataset: {data_set}')
        self.path = path
        self.data_set = data_set
        self.canonical = ALIASES[data_set]
        self.base_size = base_size
        self.crop_size = crop_size
        self.istraining = istraining
        split = 'train.txt' if istraining else 'test.txt'
        split_path = split_file or os.path.join(path, 'set_configs', self.canonical, split)
        with open(split_path, encoding='utf-8') as handle:
            ids = [line.strip() for line in handle if line.strip()]
        root = os.path.join(path, self.canonical)
        self.images, self.masks = [], []
        for item in ids:
            image_path = _resolve_file(os.path.join(root, 'images'), item)
            mask_path = _resolve_file(os.path.join(root, 'masks'), item)
            if image_path is None or mask_path is None:
                raise FileNotFoundError(f'Missing pair for {item}: {image_path}, {mask_path}')
            self.images.append(image_path)
            self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _simple_augment(image, mask):
        if random.random() < 0.5:
            image, mask = image[::-1, :, :], mask[::-1, :]
        if random.random() < 0.5:
            image, mask = image[:, ::-1, :], mask[:, ::-1]
        if random.random() < 0.5:
            image, mask = image.transpose(1, 0, 2), mask.transpose(1, 0)
        return image, mask

    def _sync_transform(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        crop = self.crop_size
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = image.size
        if h > w:
            oh, ow = long_size, int(w * long_size / h + 0.5)
        else:
            ow, oh = long_size, int(h * long_size / w + 0.5)
        image = image.resize((ow, oh), Image.Resampling.BILINEAR)
        mask = mask.resize((ow, oh), Image.Resampling.NEAREST)
        padw, padh = max(0, crop - ow), max(0, crop - oh)
        if padw or padh:
            image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        w, h = image.size
        x1, y1 = random.randint(0, w - crop), random.randint(0, h - crop)
        image = image.crop((x1, y1, x1 + crop, y1 + crop))
        mask = mask.crop((x1, y1, x1 + crop, y1 + crop))
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return image, mask

    def _normalized_tensor(self, image):
        arr = np.asarray(image, dtype=np.float32)
        if self.canonical == 'NUDT-SIRST':
            arr = (arr - 107.80905151367188) / 33.02274703979492
        else:
            arr /= 255.0
            mean, std = NORMALIZATION[self.canonical]
            arr = (arr - mean) / std
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float()

    def __getitem__(self, index):
        image_path, mask_path = self.images[index], self.masks[index]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        raw_mask = np.asarray(mask)
        mask = Image.fromarray(np.where(raw_mask >= 127, 255, 0).astype(np.uint8))

        if self.canonical == 'NUDT-SIRST':
            image = image.resize((self.base_size, self.base_size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.base_size, self.base_size), Image.Resampling.BILINEAR)
            image_arr = np.asarray(image, dtype=np.float32)
            mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
            if self.istraining:
                image_arr, mask_arr = self._simple_augment(image_arr, mask_arr)
            image = torch.from_numpy(np.ascontiguousarray(
                ((image_arr - 107.80905151367188) / 33.02274703979492).transpose(2, 0, 1))).float()
            mask = torch.from_numpy(np.ascontiguousarray(mask_arr[None])).float()
        else:
            if self.istraining:
                image, mask = self._sync_transform(image, mask)
            else:
                eval_size = self.base_size
                image = image.resize((eval_size, eval_size), Image.Resampling.BILINEAR)
                mask = mask.resize((eval_size, eval_size), Image.Resampling.NEAREST)
            image = self._normalized_tensor(image)
            mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
            mask = torch.from_numpy(np.ascontiguousarray(mask_arr[None])).float()
        return image, (mask > 0).float(), os.path.basename(image_path)

