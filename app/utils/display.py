import os
from typing import Union, Tuple, List

import matplotlib.pyplot as plt
import tensorflow as tf


def imshow(image, title=None):
    """Display an image along with its title."""
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

def show_images_with_title(images: Union[list, tuple], titles: Union[list, tuple]):
    """Displays a row of images (along with their titles)"""
    if len(images) != len(titles):
        raise ValueError(f"titles are not complete, got {titles}")

    plt.figure(figsize=(20, 12))
    for idx, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), idx + 1)
        plt.xticks([])
        plt.yticks([])
        imshow(image, title)

def load_image(image_path: str, channels: int):
    image = tf.io.read_file(image_path)

    # Get image extension
    ext = os.path.splitext(image_path)[1]
    if ext == ".png":
        decode = tf.image.decode_png
    elif ext in [".jpg", '.jpeg']:
        decode = tf.image.decode_jpeg
    elif ext == 'bmp':
        decode = tf.image.decode_bmp
    else:
        decode = tf.image.decode_image

    return decode(image, channels=channels)


def load_show_images(image_paths: Union[List, Tuple],
                     channels: Union[List, Tuple],
                     titles: Union[List, Tuple]):
    """Load and display images along with their titles."""
    images = [load_image(path, channel) for path, channel in zip(image_paths, channels)]

    show_images_with_title(images, titles)






