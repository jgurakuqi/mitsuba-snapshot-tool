import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
from cv2 import imwrite
from os import path, makedirs

# (llvm, cuda) + (mono, spectral) + (polarized)
mi.set_variant("llvm_mono_polarized")


def simulate_pfa_mosaic(
    S0, S1, S2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Compute/Simulate PFA camera mosaic pattern from Stokes parameters

    Args:
        S0 (np.ndarray): stoke s0
        S1 (np.ndarray): stoke s1
        S2 (np.ndarray): stoke s2

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Simulated channels I0, I45, I90, I135.
    """
    I0 = 0.5 * (S0 + S1)
    I45 = 0.5 * (S0 + S2)
    I90 = 0.5 * (S0 - S1)
    I135 = 0.5 * (S0 - S2)
    return I0, I45, I90, I135


def load_interpolations(file_path: str):
    """
    Load the interpolators.

    Args:
        file_path (str): path to the interpolators.

    Returns:
        function: Interpolating functions.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def create_directory(directory: str):
    """
    Create the folder in the given path if it does not exist.

    Args:
        directory (str): path to check.
    """
    try:
        if not path.exists(directory):
            print(f"Folder '{directory}' does not exists. Starting creation...")
            makedirs(directory)
            print(f"Folder '{directory}' created successfully.")
    except OSError as e:
        print(f"Error creating folder: {e}")


def check_output_folders(
    chosen_shape: str,
    output_directory: str,
    comparator_folder_name: str,
    images_folder_name: str,
    deep_shape_folder_name: str,
) -> None:
    """
    Check if the output folders for images and .mat files exist. If not, create
    the missing ones.

    Args:
        chosen_shape (str): Chosen Mitsuba shape to render. Used to name the internal folder.
        output_directory (str): Base output directory which includes all the folders of every kind
        of output.
        comparator_folder_name (str): Relative folder path for the comparator outputs.
        images_folder_name (str): Relative folder path for the output images.
        deep_shape_folder_name (str): Relative folder path for the Deep Shape Network outputs.
    """
    # *** CHECKS folders for output images. ***

    create_directory(output_directory)
    create_directory(f"{output_directory}{images_folder_name}")
    create_directory(f"{output_directory}{images_folder_name}{chosen_shape}/")

    # *** CHECKS folders for .mat outputs for Matlab comparator. ***

    comparator_folder_path = f"{output_directory}{comparator_folder_name}"
    # Check if comparator output folder exists, ...
    create_directory(comparator_folder_path)
    # Check if comparator-shape folder exists, ...
    current_scene_comparator_path = f"{comparator_folder_path}{chosen_shape}/"
    create_directory(current_scene_comparator_path)

    # *** CHECKS folders for .mat outputs for Deep Shape's Neural Network. ***

    create_directory(f"{output_directory}{deep_shape_folder_name}")
    create_directory(f"{output_directory}{deep_shape_folder_name}{chosen_shape}/")


def write_output_images(
    S0: np.ndarray,
    dolp: np.ndarray,
    angle_n: np.ndarray,
    normals: np.ndarray,
    output_directory: str,
    images_folder_name: str,
    chosen_shape: str,
    chosen_camera: str,
    chosen_material: str,
    chosen_reflectance: str,
    fov: float,
) -> None:
    """
    Save the given Polarization information as file for purposes of debug.

    Args:
        S0 (np.ndarray): Stoke 0 (i.e., total intensity).
        dolp (np.ndarray): Degree of linear polarization.
        angle_n (np.ndarray): Colourized angle of linear polarization.
        normals (np.ndarray): Ground truth surface normals.
        output_directory (str): Pathname of the top level folder which will contain all
        kinds of outputs.
        images_folder_name (str): Relative path to folder containing output images.
        chosen_shape (str): Chosen Mitsuba shape to be rendered. Used to name the internal folder to
        group the output images by shape.
        chosen_camera (str): Chosen camera type. Used for the filename.
        chosen_material (str): Chosen shape's material. Used for the filename.
        chosen_reflectance (str): Chosen shape's reflectance. Used for the filename.
        fov (int): Current fov. Used for the filename.
    """
    prefix_path = f"{output_directory}{images_folder_name}{chosen_shape}/{chosen_camera}_{chosen_material}_{chosen_reflectance}_{fov}_deg_fov_"
    S0_pathname = f"{prefix_path}S0.png"
    DOLP_pathname = f"{prefix_path}DOLP.png"
    AOLP_pathname = f"{prefix_path}AOLP.png"
    NORMALS_pathname = f"{prefix_path}NORMALS.png"

    imwrite(S0_pathname, np.clip(S0 * 255.0, 0, 255).astype(np.uint8))
    imwrite(DOLP_pathname, (dolp * 255.0).astype(np.uint8))
    imwrite(AOLP_pathname, angle_n)
    imwrite(NORMALS_pathname, ((normals + 1.0) * 127.5).astype(np.uint8))


def extract_chosen_layers_as_numpy(
    film: mi.Film, layer_name_to_px_format_dict: dict[str, mi.Bitmap.PixelFormat]
) -> dict[str, np.ndarray]:
    """
    Like the "extract_layer_as_numpy" function, but allows to extract multiple chosen layers
    at the same time, avoiding the multiple visit of the film object.

    Args:
        film (mi.Film):  The film object from which to extract the layer.
        layer_name_to_px_format_dict (dict[str, mi.Bitmap.PixelFormat]): Px format of the layers
        mapped by their name.

    Returns:
        dict[str, np.ndarray]: Contains all the required layers mapped by their name.
    """
    return {
        layer[0]: np.array(
            layer[1].convert(
                layer_name_to_px_format_dict[layer[0]],
                mi.Struct.Type.Float64,  # previously 32
                srgb_gamma=False,
            )
        )
        for layer in film.bitmap(raw=False).split()
        if layer[0] in layer_name_to_px_format_dict.keys()
    }


def plot_rgb_image(image: np.ndarray) -> None:
    """
    Plot the given RGB image.

    Args:
        image (numpy.ndarray): The RGB image as a 2D NumPy array.
    """
    plt.imshow(image)
    plt.axis("on")
    plt.show()
