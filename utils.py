import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

# (llvm, cuda) + (mono, spectral) + (polarized)
mi.set_variant("cuda_spectral_polarized")


def extract_layer_as_numpy(
    film: mi.Film, name: str, pxformat: mi.Bitmap.PixelFormat
) -> np.ndarray:
    """
    Extract a layer from the film as a NumPy array.

    Args:
        film (mitsuba.Film): The film object from which to extract the layer.
        name (str): Name of the layer to extract.
        pxformat (mitsuba.Bitmap.PixelFormat): Pixel format for the extracted layer.

    Raises:
        ValueError: Thrown if the required layer is not found.

    Returns:
        np.ndarray: The extracted layer as a NumPy array.
    """
    for layer in film.bitmap(raw=False).split():
        if layer[0] == name:
            return np.array(
                layer[1].convert(pxformat, mi.Struct.Type.Float32, srgb_gamma=False)
            )
    raise ValueError(f"[extract_layer_as_numpy]: Layer -- {name} -- not found")


def plot_rgb_image(image: np.ndarray) -> None:
    """
    Plot the RGB image.

    Args:
        image (numpy.ndarray): The RGB image as a 2D NumPy array.
    """
    # print(set(image.flatten()))
    plt.imshow(image)
    plt.axis("on")
    plt.show()
