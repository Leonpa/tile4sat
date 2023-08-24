"""
Helper file that is used in img_size_testing.py to convert png images to triplets of tiles
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt


def get_tile_png(tile_dir, img_dir, tiles_per_img=1, tile_size=50, val_type='uint8', bands_only=False, verbose=False):
    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir)
    size_even = (tile_size % 2 == 0)
    tile_radius = tile_size // 2
    for img_name in os.listdir(img_dir):
        if img_name[-3:] == 'png':
            if verbose:
                print("Sampling image {}".format(img_name))
            img = load_img_matplotlib(os.path.join(img_dir, img_name), val_type=val_type, bands_only=bands_only)
            img_height = img.shape[0]
            img_width = img.shape[1]
            for i in range(tiles_per_img):
                x = random.randint(tile_radius, img_width - tile_radius - 1)
                y = random.randint(tile_radius, img_height - tile_radius - 1)
                tile = img[y - tile_radius:y + tile_radius + size_even, x - tile_radius:x + tile_radius + size_even, :]
                tile_filename = "{}_tile{}.npy".format(os.path.splitext(img_name)[0], i+1)  # Use image name as prefix
                np.save(os.path.join(tile_dir, tile_filename), tile)


def load_img_matplotlib(img_file, val_type='uint8', bands_only=True, num_bands=3):
    """
    Loads an image using matplotlib, returns it as an array.
    """
    img = plt.imread(img_file)

    # Exclude the bottom 14 pixels of the google watermark
    img = img[:-14, :, :]

    if val_type == 'uint8':
        img = (img * 255).astype(np.uint8)
    elif val_type == 'float32':
        pass
    else:
        raise ValueError('Invalid val_type for image values. Try uint8 or float32.')

    # Ensure that grayscale images have a single channel
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    if bands_only and img.shape[2] > num_bands:
        img = img[:, :, :num_bands]

    return img


def group_and_display(array, tiles_per_img, img_dir, n_img_class):
    """
    Groups tiles by their image index and displays the original image and associated class labels.

    Parameters:
        - array: List of [image_index, label] pairs
        - tiles_per_img: Number of tiles per image
        - img_dir: Directory containing the original images
    """
    grouped = {}
    for item in array:
        img_index, label = item
        if img_index not in grouped:
            grouped[img_index] = []
        grouped[img_index].append(label)

    # Display each original image and its associated class labels
    for img_index, labels in grouped.items():
        if img_index == n_img_class:
            break
        if len(labels) == tiles_per_img:  # Only display if we have the expected number of tiles
            # Load and display the original image
            img_path = os.path.join(img_dir, "{}.png".format(img_index))
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                plt.imshow(img)
                plt.title("Image Index: {} - Labels: {}".format(img_index, labels))
                plt.axis('off')
                plt.show()
            else:
                print(f"Image {img_index}.png not found in the directory.")


def filter_and_plot_pure_classes(array, img_dir, cat=1, n=None):
    """
    Filters images with pure class sequences and displays them side by side in sequential order.

    Parameters:
        - array: List of [image_index, label] pairs
        - img_dir: Directory containing the original images
        - cat: Optional, the category/class to filter images by
    """

    # Group tiles by their image index
    grouped = {}
    for item in array:
        img_index, label = item
        if not img_index.isdigit():  # Skip non-numeric image indices
            continue
        if img_index not in grouped:
            grouped[img_index] = []
        grouped[img_index].append(label)

    # Filter out groups with pure class sequences and optionally by the provided category
    pure_sequences = []
    for img_index, labels in grouped.items():
        if all(label == labels[0] for label in labels) and (cat is None or labels[0] == cat):
            pure_sequences.append((img_index, labels[0]))

    # Sort pure_sequences based on image indices for sequential plotting
    pure_sequences.sort(key=lambda x: int(x[0]))

    # Plot images with pure class sequences
    i = 0
    while i < len(pure_sequences) - 2:  # Iterate in steps of 3
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        for j in range(3):
            img_path = os.path.join(img_dir, "{}.png".format(pure_sequences[i + j][0]))
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                ax[j].imshow(img)
                ax[j].set_title(f"Image Index: {pure_sequences[i + j][0]} - Class: {pure_sequences[i + j][1]}")
                ax[j].axis('off')
            else:
                ax[j].set_title(f"Image {pure_sequences[i + j][0]}.png not found")
                ax[j].axis('off')

        plt.tight_layout()
        plt.show()
        i += 3
        if i >= n:
            break
