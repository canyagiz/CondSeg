"""
CondSeg Dataset with Albumentations Augmentation + Smart Fill
=============================================================
Geometric augmentations (ShiftScaleRotate, ElasticTransform) boş
bıraktıkları alanları siyahla değil, kenar piksellerin blur+fade
gradyanıyla doldurur — ten rengine uyumlu, doğal görünüm.

Ek classlar:
  SmartFillShiftScaleRotate  — SSR + gradient void fill
  SmartFillElasticTransform  — ElasticTransform + gradient void fill
  _gradient_fill_void        — ortak yardımcı fonksiyon
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import cv2

import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch import ToTensorV2


# ──────────────────────────────────────────────────────────────
# Yardımcı: Boş (siyah) pikselleri gradient ile doldur
# ──────────────────────────────────────────────────────────────

def _gradient_fill_void(image: np.ndarray, void_mask: np.ndarray) -> np.ndarray:
    """
    void_mask: (H, W) bool — True olan pikseller boş/doldurulacak.
    image    : (H, W, 3) uint8

    Her boş sütun/satır için en yakın geçerli pikselden
    Gaussian-blur + renk-fade uygular; process_image_gradient_fade
    mantığına benzer ama 2D void bölgeleri için genelleştirilmiş.
    """
    if not void_mask.any():
        return image

    result = image.copy().astype(np.float64)
    H, W = void_mask.shape

    # ---- Geçerli piksellerin medyan rengi (global fallback) ----
    valid_pixels = image[~void_mask]           # (N, 3)
    if len(valid_pixels) == 0:
        return image
    global_color = np.median(valid_pixels, axis=0).astype(np.float64)  # [R,G,B]

    # ---- Mesafe haritası: boş pikselin en yakın geçerli piksele uzaklığı ----
    # cv2.distanceTransform: void bölgelerin içindeki uzaklık
    void_u8 = void_mask.astype(np.uint8)
    dist_map = cv2.distanceTransform(void_u8, cv2.DIST_L2, 5)  # (H, W) float32
    max_dist = dist_map.max()
    if max_dist == 0:
        return image

    # ---- Geçerli piksellerden yayılmış yumuşatılmış görüntü ----
    # Strateji: boş pikselleri önce inpainting ile doldur
    # (fast_marching / telea yöntemi — kenar değerlerinden yayılır)
    inpainted = cv2.inpaint(image, void_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # inpainted: (H, W, 3) — kenar renklerinden yayılmış, ama keskin

    # ---- Mesafeye göre blur + global_color fade uygula ----
    # t = 0 (kenara yapışık) → inpainted rengi
    # t = 1 (en uzak)        → global_color (düz ton)
    t = np.clip(dist_map / max_dist, 0.0, 1.0)           # (H, W)
    alpha = t * t * (3.0 - 2.0 * t)                       # smoothstep

    # Artan blur: yakında az, uzakta çok
    # Birkaç blur seviyesi üret, mesafeye göre interpolasyon yap
    blurred_light  = cv2.GaussianBlur(inpainted, (5,  5),  2.0).astype(np.float64)
    blurred_medium = cv2.GaussianBlur(inpainted, (21, 21), 8.0).astype(np.float64)
    blurred_heavy  = cv2.GaussianBlur(inpainted, (61, 61), 25.0).astype(np.float64)

    t3d = alpha[:, :, np.newaxis]  # (H, W, 1) — broadcast için

    # Yakın → orta → uzak: light → heavy blur → global_color
    phase1 = np.clip(t3d * 2.0, 0.0, 1.0)          # 0→0.5 arası: light→medium
    phase2 = np.clip(t3d * 2.0 - 1.0, 0.0, 1.0)    # 0.5→1.0 arası: medium→color

    blurred_interp = blurred_light * (1 - phase1) + blurred_medium * phase1
    final_fill     = blurred_interp * (1 - phase2) + global_color * phase2

    # Sadece void bölgelere yaz
    void3d = void_mask[:, :, np.newaxis]
    result = np.where(void3d, final_fill, result)

    return np.clip(result, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────
# SmartFillShiftScaleRotate
# ──────────────────────────────────────────────────────────────

class SmartFillShiftScaleRotate(DualTransform):
    """
    ShiftScaleRotate gibi çalışır ama boş alanları siyahla değil
    gradient fill ile doldurur (sadece image; mask siyah kalır).

    Parametreler standart ShiftScaleRotate ile aynı.
    """

    def __init__(
        self,
        shift_limit=0.05,
        scale_limit=0.15,
        rotate_limit=15,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.shift_limit   = shift_limit
        self.scale_limit   = scale_limit
        self.rotate_limit  = rotate_limit
        self.interpolation = interpolation
        self.border_mode   = border_mode

    def get_params(self):
        return {
            "angle": np.random.uniform(-self.rotate_limit, self.rotate_limit),
            "scale": np.random.uniform(1 - self.scale_limit, 1 + self.scale_limit),
            "dx":    np.random.uniform(-self.shift_limit,  self.shift_limit),
            "dy":    np.random.uniform(-self.shift_limit,  self.shift_limit),
        }

    def _get_warp_matrix(self, h, w, angle, scale, dx, dy):
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        M[0, 2] += dx * w
        M[1, 2] += dy * h
        return M

    def apply(self, image, angle=0, scale=1, dx=0, dy=0, **params):
        """Görüntüye uygula + void bölgeleri gradient fill et."""
        h, w = image.shape[:2]
        M = self._get_warp_matrix(h, w, angle, scale, dx, dy)

        # Void tespiti: orijinal alpha kanalı gibi bir "maske" warpla
        probe = np.ones((h, w), dtype=np.uint8) * 255
        probe_warped = cv2.warpAffine(
            probe, M, (w, h),
            flags=self.interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        void_mask = probe_warped < 128  # True = boş alan

        # Görüntüyü warpla (şimdilik constant=0)
        warped = cv2.warpAffine(
            image, M, (w, h),
            flags=self.interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Boş alanları gradient ile doldur
        if void_mask.any():
            warped = _gradient_fill_void(warped, void_mask)

        return warped

    def apply_to_mask(self, mask, angle=0, scale=1, dx=0, dy=0, **params):
        """Mask için normal warp — sınır değeri 0 kalır (label bozulmasın)."""
        h, w = mask.shape[:2]
        M = self._get_warp_matrix(h, w, angle, scale, dx, dy)
        return cv2.warpAffine(
            mask, M, (w, h),
            flags=cv2.INTER_NEAREST,   
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def get_transform_init_args_names(self):
        return ("shift_limit", "scale_limit", "rotate_limit", "interpolation", "border_mode")


# ──────────────────────────────────────────────────────────────
# SmartFillElasticTransform
# ──────────────────────────────────────────────────────────────

class SmartFillElasticTransform(DualTransform):
    """
    ElasticTransform + gradient void fill.
    ElasticTransform çok az void üretir ama yine de tutarlı olsun.
    """

    def __init__(
        self,
        alpha=30,
        sigma=5,
        always_apply=False,
        p=0.15,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.alpha = alpha
        self.sigma = sigma

    def get_params(self):
        seed = np.random.randint(0, 2**31)
        return {"seed": seed}

    def _get_flow(self, h, w, seed):
        rng = np.random.default_rng(seed)
        dx = cv2.GaussianBlur(
            (rng.random((h, w)) * 2 - 1).astype(np.float32),
            (0, 0), self.sigma
        ) * self.alpha
        dy = cv2.GaussianBlur(
            (rng.random((h, w)) * 2 - 1).astype(np.float32),
            (0, 0), self.sigma
        ) * self.alpha
        return dx, dy

    def _remap(self, src, dx, dy, interp):
        h, w = src.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)
        return cv2.remap(
            src, map_x, map_y,
            interpolation=interp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def apply(self, image, seed=0, **params):
        h, w = image.shape[:2]
        dx, dy = self._get_flow(h, w, seed)

        # Void tespiti
        probe = np.ones((h, w), dtype=np.uint8) * 255
        probe_warped = self._remap(probe, dx, dy, cv2.INTER_LINEAR)
        void_mask = probe_warped < 128

        warped = self._remap(image, dx, dy, cv2.INTER_LINEAR)
        if void_mask.any():
            warped = _gradient_fill_void(warped, void_mask)
        return warped

    def apply_to_mask(self, mask, seed=0, **params):
        h, w = mask.shape[:2]
        dx, dy = self._get_flow(h, w, seed)
        return self._remap(mask, dx, dy, cv2.INTER_NEAREST)

    def get_transform_init_args_names(self):
        return ("alpha", "sigma")


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class EyeSegmentationDataset(Dataset):
    """
    Dataset for CondSeg training with Albumentations augmentation.

    Geometric augmentations boş alanları siyahla değil ten rengine
    uyumlu gradient fill ile doldurur.

    Directory structure:
        data/<split>/images/*.jpg
        data/<split>/masks/*.png     (single-channel, values {0,1,2,3})

    Classes:
        0: Background  |  1: Sclera  |  2: Iris  |  3: Pupil

    Derived binary masks:
        GT_Eye_Region    = (mask >= 1)
        GT_Visible_Iris  = (mask == 2) | (mask == 3)
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, data_root: str, img_size: int = 1024, augment: bool = False):
        super().__init__()
        self.img_size = img_size
        self.augment  = augment

        image_dir = os.path.join(data_root, "images")
        mask_dir  = os.path.join(data_root, "masks")

        image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        mask_files_map = {
            os.path.splitext(os.path.basename(mf))[0]: mf
            for mf in glob.glob(os.path.join(mask_dir, "*.png"))
        }

        self.pairs = [
            (img, mask_files_map[base])
            for img in image_files
            if (base := os.path.splitext(os.path.basename(img))[0]) in mask_files_map
        ]

        assert len(self.pairs) > 0, f"No image-mask pairs found in {data_root}"
        print(f"  Dataset: {len(self.pairs)} pairs from {data_root} (augment={augment})")

        # ---- Augmentation pipeline ----
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),

                # === Geometric ===
                A.HorizontalFlip(p=0.5),

                # ↓ Siyah dolgu YOK — gradient fill kullanır
                SmartFillShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.15,
                    rotate_limit=15,
                    interpolation=cv2.INTER_LINEAR,
                    p=0.7,
                ),
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=cv2.INTER_LINEAR,
                    p=0.5,
                ),
                # ↓ Siyah dolgu YOK — gradient fill kullanır
                SmartFillElasticTransform(
                    alpha=30,
                    sigma=5,
                    p=0.3,
                ),

                # === Photometric (image only) ===
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
                A.RandomGamma(gamma_limit=(70, 150), p=0.4),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 15), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ], p=0.5),
                A.GaussNoise(std_range=(0.01, 0.04), p=0.4),
                A.ToGray(p=0.1),

                # === Normalize + tensor ===
                A.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """
        Returns:
            image           : (3, H, W) float32
            gt_eye_region   : (1, H, W) float32
            gt_visible_iris : (1, H, W) float32
        """
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        result = self.transform(image=image, mask=mask)
        image  = result["image"]    # (3, H, W) float32 tensor
        mask   = result["mask"]     # (H, W) uint8 numpy

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        gt_eye_region   = (mask >= 1).float().unsqueeze(0)
        gt_visible_iris = ((mask == 2) | (mask == 3)).float().unsqueeze(0)

        return image, gt_eye_region, gt_visible_iris


# ──────────────────────────────────────────────────────────────
# Test yardımcısı
# ──────────────────────────────────────────────────────────────

def create_dummy_gt_masks(batch_size: int, H: int, W: int, device: str = "cpu"):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    dist = torch.sqrt(xx ** 2 + yy ** 2).unsqueeze(0).expand(batch_size, -1, -1)
    mask = torch.zeros(batch_size, H, W, dtype=torch.long, device=device)
    mask[dist < 0.8] = 1
    mask[dist < 0.5] = 2
    mask[dist < 0.2] = 3
    gt_eye_region   = (mask >= 1).float().unsqueeze(1)
    gt_visible_iris = ((mask == 2) | (mask == 3)).float().unsqueeze(1)
    return gt_eye_region, gt_visible_iris