"""Training, evaluation, and inference utilities for oil-spill masks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class DetectionConfig:
    project_root: Path
    data_root: Path | None = None
    weights_path: Path | None = None
    output_dir: Path | None = None
    threshold: float = 0.5
    min_area_px: int = 30
    geo_bounds: tuple[float, float, float, float] = (72.5, 18.8, 72.85, 19.15)

    def resolved_data_root(self) -> Path:
        return (self.data_root or self.project_root / "data" / "satellite").resolve()

    def resolved_weights_path(self) -> Path:
        return (self.weights_path or self.project_root / "modules" / "detection" / "best_oil_spill_unet.pth").resolve()

    def resolved_output_dir(self) -> Path:
        return (self.output_dir or self.project_root / "outputs" / "spill").resolve()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _split_paths(config: DetectionConfig, split: str) -> tuple[list[Path], list[Path]]:
    root = config.resolved_data_root()
    image_dir = root / "images" / "images" / split
    mask_dir = root / "masks" / "masks" / split
    images = {path.stem: path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    masks = {path.stem: path for path in mask_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    missing_masks = sorted(set(images) - set(masks))
    if missing_masks:
        print(f"Warning: skipped {len(missing_masks)} {split} images without masks")
    keys = sorted(set(images) & set(masks))
    if not keys:
        raise FileNotFoundError(f"No paired {split} images and masks found under {root}")
    return [images[key] for key in keys], [masks[key] for key in keys]


class SAROilDataset(Dataset):
    def __init__(self, image_paths: list[Path], mask_paths: list[Path], transform: Any = None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.image_paths[index]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_paths[index]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Could not read pair: {self.image_paths[index]}, {self.mask_paths[index]}")
        mask = (mask > 127).astype(np.float32)
        transformed = self.transform(image=image, mask=mask) if self.transform else {"image": image, "mask": mask}
        image = transformed["image"]
        mask = transformed["mask"]
        if not torch.is_tensor(image):
            image = torch.from_numpy(image.astype(np.float32) / 255).unsqueeze(0)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return image.float(), mask.float()


def _transform(training: bool) -> A.Compose:
    operations = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)] if training else []
    operations += [A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()]
    return A.Compose(operations)


def _load_model_weights(model: nn.Module, weights_path: Path | str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint format loaded from {weights_path}")
    model.load_state_dict(state_dict)
    return model


def build_model(device: torch.device | None = None, weights_path: Path | str | None = None) -> tuple[nn.Module, torch.device]:
    device = device or select_device()
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=1, classes=1).to(device)
    if weights_path is not None:
        _load_model_weights(model, weights_path, device)
    return model, device


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits).reshape(-1)
        targets = targets.reshape(-1)
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def _metrics(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    predictions = predictions.reshape(predictions.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    intersection = (predictions * targets).sum(dim=1)
    union = (predictions + targets).sum(dim=1) - intersection
    dice = (2 * intersection + 1e-6) / (predictions.sum(dim=1) + targets.sum(dim=1) + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return float(dice.mean()), float(iou.mean())


def train_model(config: DetectionConfig, epochs: int = 5, batch_size: int = 16, learning_rate: float = 1e-3) -> Path:
    train_images, train_masks = _split_paths(config, "train")
    val_images, val_masks = _split_paths(config, "val")
    train_loader = DataLoader(SAROilDataset(train_images, train_masks, _transform(True)), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SAROilDataset(val_images, val_masks, _transform(False)), batch_size=batch_size, shuffle=False, num_workers=0)
    model, device = build_model()
    criterion = BCEDiceLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    output = config.resolved_weights_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        dices: list[float] = []
        ious: list[float] = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_loss += criterion(logits, masks).item() * images.size(0)
                dice, iou = _metrics((torch.sigmoid(logits) > config.threshold).float(), masks)
                dices.append(dice)
                ious.append(iou)
        mean_dice, mean_iou = float(np.mean(dices)), float(np.mean(ious))
        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss / len(train_loader.dataset):.4f} "
              f"val_loss={val_loss / len(val_loader.dataset):.4f} dice={mean_dice:.4f} iou={mean_iou:.4f}")
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), output)
    return output


def evaluate_model(config: DetectionConfig, split: str = "val", limit: int | None = None) -> dict[str, float]:
    image_paths, mask_paths = _split_paths(config, split)
    pairs = list(zip(image_paths, mask_paths))[:limit]
    model, device = build_model(weights_path=config.resolved_weights_path())
    model.eval()
    dices: list[float] = []
    ious: list[float] = []
    transform = _transform(False)
    with torch.no_grad():
        for image_path, mask_path in pairs:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
            transformed = transform(image=image, mask=mask)
            logits = model(transformed["image"].unsqueeze(0).to(device))
            dice, iou = _metrics((torch.sigmoid(logits) > config.threshold).float().cpu(), transformed["mask"].unsqueeze(0))
            dices.append(dice)
            ious.append(iou)
    result = {"count": len(pairs), "mean_dice": float(np.mean(dices)), "mean_iou": float(np.mean(ious))}
    print(json.dumps(result, indent=2))
    return result


def _pixel_to_geo(x: int, y: int, bounds: tuple[float, float, float, float], width: int, height: int) -> list[float]:
    min_lon, min_lat, max_lon, max_lat = bounds
    return [round(min_lon + x / width * (max_lon - min_lon), 6), round(max_lat - y / height * (max_lat - min_lat), 6)]


def detect_image(image_path: Path, model: nn.Module, device: torch.device, config: DetectionConfig, image_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    raw_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw_image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = raw_image.shape
    transformed = _transform(False)(image=raw_image)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(transformed["image"].unsqueeze(0).to(device))).squeeze().cpu().numpy()
    binary_mask = (probabilities > config.threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) >= config.min_area_px]
    detected = bool(valid_contours)
    confidence = float(np.mean(probabilities[binary_mask == 1])) if binary_mask.any() else 0.0
    bbox: list[int] = []
    features: list[dict[str, Any]] = []
    if detected:
        largest = max(valid_contours, key=cv2.contourArea)
        x, y, box_width, box_height = cv2.boundingRect(largest)
        bbox = [x, y, x + box_width, y + box_height]
        for contour in valid_contours:
            approximation = cv2.approxPolyDP(contour, 0.005 * cv2.arcLength(contour, True), True)
            polygon = [_pixel_to_geo(int(point[0][0]), int(point[0][1]), config.geo_bounds, width, height) for point in approximation]
            if len(polygon) >= 3:
                polygon.append(polygon[0])
                features.append({"type": "Feature", "properties": {"spill_id": image_id or image_path.stem, "confidence": round(confidence, 2)}, "geometry": {"type": "Polygon", "coordinates": [polygon]}})
    record = {"image_id": image_id or image_path.stem, "detected": detected, "confidence": round(confidence, 2), "mask_path": "", "bbox": bbox}
    geojson = {"type": "FeatureCollection", "features": features}
    return record, geojson, binary_mask * 255


def infer_directory(config: DetectionConfig, image_dir: Path, limit: int | None = None) -> list[dict[str, Any]]:
    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    model, device = build_model(weights_path=config.resolved_weights_path())
    model.eval()
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)[:limit]
    records = []
    for image_path in images:
        record, geojson, mask = detect_image(image_path, model, device, config)
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        geojson_path = output_dir / f"{image_path.stem}.geojson"
        cv2.imwrite(str(mask_path), mask)
        geojson_path.write_text(json.dumps(geojson, indent=2))
        record["mask_path"] = str(mask_path.relative_to(config.project_root))
        (output_dir / f"{image_path.stem}.json").write_text(json.dumps(record, indent=2))
        records.append(record)
    (output_dir / "all_detections.json").write_text(json.dumps(records, indent=2))
    print(f"Processed {len(records)} images into {output_dir}")
    return records


def _config_from_args(args: argparse.Namespace) -> DetectionConfig:
    root = Path(args.project_root).resolve()
    bounds = tuple(float(value) for value in args.geo_bounds.split(","))
    if len(bounds) != 4:
        raise ValueError("geo-bounds must contain four comma-separated values")
    return DetectionConfig(root, weights_path=Path(args.weights).resolve() if args.weights else None, output_dir=Path(args.output).resolve() if args.output else None, threshold=args.threshold, min_area_px=args.min_area, geo_bounds=bounds)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "infer"))
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[2])
    parser.add_argument("--weights")
    parser.add_argument("--output")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--geo-bounds", default="72.5,18.8,72.85,19.15")
    parser.add_argument("--split", default="val")
    parser.add_argument("--image-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config = _config_from_args(args)
    if args.command == "train":
        print(f"Saved best model to {train_model(config, args.epochs, args.batch_size)}")
    elif args.command == "evaluate":
        evaluate_model(config, args.split, args.limit)
    else:
        image_dir = Path(args.image_dir) if args.image_dir else config.resolved_data_root() / "images" / "images" / args.split
        infer_directory(config, image_dir, args.limit)


if __name__ == "__main__":
    main()