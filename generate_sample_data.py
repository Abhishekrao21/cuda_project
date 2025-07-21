#!/usr/bin/env python3
"""
Sample Data Generation Script for CUDA Histogram Project
Generates a mix of small and large synthetic images for testing
"""

import numpy as np
import cv2
import os
import random
from pathlib import Path

def create_synthetic_image(width, height, pattern_type='random'):
    """Create a synthetic image with specified pattern"""

    if pattern_type == 'random':
        # Random noise image
        image = np.random.randint(0, 256, (height, width), dtype=np.uint8)

    elif pattern_type == 'gradient':
        # Gradient image
        x_grad = np.linspace(0, 255, width, dtype=np.uint8)
        image = np.tile(x_grad, (height, 1))

    elif pattern_type == 'checkerboard':
        # Checkerboard pattern
        square_size = min(width, height) // 8
        image = np.zeros((height, width), dtype=np.uint8)
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    end_i = min(i + square_size, height)
                    end_j = min(j + square_size, width)
                    image[i:end_i, j:end_j] = 255

    elif pattern_type == 'circles':
        # Concentric circles
        image = np.zeros((height, width), dtype=np.uint8)
        center_x, center_y = width // 2, height // 2
        max_radius = min(width, height) // 2

        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        for i in range(0, max_radius, max_radius // 5):
            mask = (distances >= i) & (distances < i + max_radius // 10)
            image[mask] = (i * 255 // max_radius)

    elif pattern_type == 'noise_bands':
        # Horizontal bands with different noise levels
        image = np.zeros((height, width), dtype=np.uint8)
        band_height = height // 4

        for i in range(4):
            start_y = i * band_height
            end_y = min((i + 1) * band_height, height)
            noise_level = (i + 1) * 60
            band = np.random.randint(0, noise_level, (end_y - start_y, width), dtype=np.uint8)
            image[start_y:end_y, :] = band

    elif pattern_type == 'gaussian_blobs':
        # Random Gaussian blobs
        image = np.zeros((height, width), dtype=np.uint8)
        num_blobs = random.randint(3, 8)

        for _ in range(num_blobs):
            center_x = random.randint(width // 4, 3 * width // 4)
            center_y = random.randint(height // 4, 3 * height // 4)
            sigma_x = random.randint(width // 10, width // 4)
            sigma_y = random.randint(height // 10, height // 4)
            intensity = random.randint(100, 255)

            y, x = np.ogrid[:height, :width]
            blob = intensity * np.exp(-((x - center_x)**2 / (2 * sigma_x**2) + 
                                     (y - center_y)**2 / (2 * sigma_y**2)))
            image = np.maximum(image, blob.astype(np.uint8))

    else:  # default to random
        image = np.random.randint(0, 256, (height, width), dtype=np.uint8)

    return image

def generate_test_dataset(output_dir, num_small=15, num_large=10):
    """Generate a mixed dataset of small and large images"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    patterns = ['random', 'gradient', 'checkerboard', 'circles', 'noise_bands', 'gaussian_blobs']

    print(f"Generating test dataset in: {output_dir}")
    print(f"Small images: {num_small}, Large images: {num_large}")

    # Generate small images (64x64 to 256x256)
    for i in range(num_small):
        width = random.choice([64, 128, 192, 256])
        height = random.choice([64, 128, 192, 256])
        pattern = random.choice(patterns)

        image = create_synthetic_image(width, height, pattern)
        filename = f"small_{i:03d}_{pattern}_{width}x{height}.png"
        filepath = output_path / filename

        cv2.imwrite(str(filepath), image)
        print(f"Created: {filename}")

    # Generate large images (512x512 to 1024x1024)
    for i in range(num_large):
        width = random.choice([512, 768, 1024])
        height = random.choice([512, 768, 1024])
        pattern = random.choice(patterns)

        image = create_synthetic_image(width, height, pattern)
        filename = f"large_{i:03d}_{pattern}_{width}x{height}.png"
        filepath = output_path / filename

        cv2.imwrite(str(filepath), image)
        print(f"Created: {filename}")

    # Generate some edge cases
    # Very small image
    tiny_image = create_synthetic_image(32, 32, 'gradient')
    cv2.imwrite(str(output_path / "tiny_001_gradient_32x32.png"), tiny_image)

    # Very large image (if system can handle it)
    try:
        huge_image = create_synthetic_image(2048, 1536, 'circles')
        cv2.imwrite(str(output_path / "huge_001_circles_2048x1536.png"), huge_image)
        print("Created: huge_001_circles_2048x1536.png")
    except MemoryError:
        print("Skipped huge image due to memory constraints")

    # High contrast image
    high_contrast = np.zeros((256, 256), dtype=np.uint8)
    high_contrast[:128, :] = 255  # Half white, half black
    cv2.imwrite(str(output_path / "contrast_001_binary_256x256.png"), high_contrast)

    # Low contrast image
    low_contrast = np.full((256, 256), 128, dtype=np.uint8)
    noise = np.random.randint(-10, 11, (256, 256))
    low_contrast = np.clip(low_contrast + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path / "contrast_002_lowcontrast_256x256.png"), low_contrast)

    print(f"\nDataset generation complete!")
    print(f"Total images created: {num_small + num_large + 4}")

    # Generate a summary file
    summary_file = output_path / "dataset_info.txt"
    with open(summary_file, 'w') as f:
        f.write("CUDA Histogram Test Dataset Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Small images (64x64 to 256x256): {num_small}\n")
        f.write(f"Large images (512x512 to 1024x1024): {num_large}\n")
        f.write("Special test cases: 4\n")
        f.write(f"  - Tiny image (32x32)\n")
        f.write(f"  - Huge image (2048x1536) - if generated\n")
        f.write(f"  - High contrast binary image\n")
        f.write(f"  - Low contrast noisy image\n\n")
        f.write("Pattern types used:\n")
        for pattern in patterns:
            f.write(f"  - {pattern}\n")
        f.write("\nThis dataset is designed to test:\n")
        f.write("  - Mixed image sizes and batch processing\n")
        f.write("  - Different histogram distributions\n")
        f.write("  - Edge cases (very small/large images)\n")
        f.write("  - Contrast variations\n")

    print(f"Dataset summary saved to: {summary_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate test dataset for CUDA histogram project")
    parser.add_argument("--output", "-o", default="data", 
                       help="Output directory for generated images")
    parser.add_argument("--small", "-s", type=int, default=15,
                       help="Number of small images to generate")
    parser.add_argument("--large", "-l", type=int, default=10, 
                       help="Number of large images to generate")

    args = parser.parse_args()

    generate_test_dataset(args.output, args.small, args.large)

if __name__ == "__main__":
    main()
