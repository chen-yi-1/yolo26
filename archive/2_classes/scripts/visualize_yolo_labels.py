#!/usr/bin/env python3
"""Visualize YOLO segmentation labels on images for manual verification."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_yolo_segmentation(label_path, image_width, image_height):
    """Parse YOLO segmentation label file.
    
    Format: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates)
    
    Returns: list of (class_id, polygon_points) tuples
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    
    labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue  # Need at least class_id + 3 points (6 coords)
            
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            # Convert normalized coordinates to pixel coordinates
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * image_width)
                y = int(coords[i + 1] * image_height)
                points.append((x, y))
            
            labels.append((class_id, points))
    
    return labels


def draw_labels(image, labels, class_names=None):
    """Draw segmentation masks and bounding boxes on image."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Generate distinct colors for different classes
    colors = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255),    # blue
        (255, 255, 0),  # yellow
        (255, 0, 255),  # magenta
        (0, 255, 255),  # cyan
        (255, 128, 0),  # orange
        (128, 0, 255),  # purple
        (0, 128, 255),  # sky blue
        (128, 255, 0),  # lime
    ]
    
    for class_id, points in labels:
        color = colors[class_id % len(colors)]
        
        # Draw segmentation mask (semi-transparent)
        mask = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(points, fill=(color[0], color[1], color[2], 64))
        
        # Composite mask onto image
        image = Image.alpha_composite(image.convert('RGBA'), mask).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Draw polygon outline
        draw.polygon(points, outline=color, width=2)
        
        # Draw bounding box
        points_array = np.array(points)
        x_min, y_min = points_array.min(axis=0)
        x_max, y_max = points_array.max(axis=0)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=1)
        
        # Draw class label
        if class_names is not None and class_id < len(class_names):
            label_text = class_names[class_id]
        else:
            label_text = f"class {class_id}"
        
        # Draw label background (PIL 10+ compatible)
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), label_text)
            text_width = right - left
            text_height = bottom - top
        else:
            text_width, text_height = draw.textsize(label_text)
        label_x = x_min
        label_y = max(y_min - text_height - 4, 0)
        draw.rectangle([label_x, label_y, label_x + text_width + 4, label_y + text_height + 4], fill=color)
        draw.text((label_x + 2, label_y + 2), label_text, fill=(255, 255, 255))
    
    return image


def visualize_labels(image_path, label_path, class_names=None):
    """Load image, parse labels, draw and display."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Parse labels
    labels = parse_yolo_segmentation(label_path, width, height)
    
    # Draw labels
    result = draw_labels(image, labels, class_names)
    
    # Show image
    result.show(title=f"YOLO Labels: {Path(image_path).name}")
    
    return labels


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO segmentation labels on images for verification."
    )
    parser.add_argument('--image_path', default=r"C:\Users\EDY\Desktop\public_datas\abnormal\images\microgreen_2026-01-30_15-39-10_jpg.rf.be56cc285980dd657ff9f5e1c77444bc.jpg", help='Path to the image file')
    parser.add_argument('--label_path', default=r"C:\Users\EDY\Desktop\public_datas\abnormal\labels\microgreen_2026-01-30_15-39-10_jpg.rf.be56cc285980dd657ff9f5e1c77444bc.txt", help='Path to the YOLO label txt file')
    parser.add_argument('--classes', default=r"C:\Users\EDY\Desktop\public_datas\data.yaml", help='Path to classes.txt or dataset.yaml file')
    return parser.parse_args()


def load_class_names(classes_path):
    """Load class names from txt or yaml file."""
    if not classes_path:
        return None
    
    classes_path = Path(classes_path)
    if not classes_path.exists():
        print(f"Warning: Classes file not found: {classes_path}")
        return None
    
    if classes_path.suffix in ('.yaml', '.yml'):
        import yaml
        with open(classes_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        names = data.get('names', {})
        if isinstance(names, dict):
            return [names[i] for i in range(len(names))]
        elif isinstance(names, list):
            return names
        else:
            return None
    else:
        with open(classes_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()
    
    image_path = Path(args.image_path)
    label_path = Path(args.label_path)
    
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    if not label_path.exists():
        print(f"Warning: Label file not found: {label_path}")
        print("Showing original image without labels...")
        image = Image.open(image_path).convert('RGB')
        image.show(title=f"Original: {image_path.name}")
        return
    
    class_names = load_class_names(args.classes)
    
    labels = visualize_labels(args.image_path, args.label_path, class_names)
    print(f"Loaded {len(labels)} labels from {label_path}")


if __name__ == "__main__":
    main()