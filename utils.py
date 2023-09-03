import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
from cv2 import imwrite
from os import path, makedirs

# (llvm, cuda) + (mono, spectral) + (polarized)
mi.set_variant("llvm_mono_polarized")


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
    angle: float,
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
        angle (int): Current view angle. Used for the filename.
    """
    prefix_path = f"{output_directory}{images_folder_name}{chosen_shape}/{chosen_camera}_{chosen_material}_{chosen_reflectance}_{angle}_"
    # 127.5 = 250 * 0.5
    imwrite(f"{prefix_path}S0.png", np.clip(S0 * 255.0, 0, 255).astype(np.uint8))
    imwrite(f"{prefix_path}DOLP.png", (dolp * 255.0).astype(np.uint8))
    imwrite(f"{prefix_path}AOLP_COLOURED.png", angle_n)
    imwrite(f"{prefix_path}NORMALS.png", ((normals + 1.0) * 127.5).astype(np.uint8))


def extract_layer_as_numpy(
    film: mi.Film, name: str, pxformat: mi.Bitmap.PixelFormat
) -> np.ndarray:
    """
    Extract a layer from the film as a NumPy array.

    Args:
        film (mi.Film): The film object from which to extract the layer.
        name (str): Name of the layer to extract.
        pxformat (mi.Bitmap.PixelFormat): Pixel format for the extracted layer.

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


# # Channels 1,2,3 = Diffuse solution
# # Channels 4,5,6 = 1st Specular Solution
# # Channels 7,8,9 = 2nd Specular Solution.
# def compute_priors(
#     aolp: np.ndarray, dolp: np.ndarray
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Calculate the physical priors based on Angle of Linear Polarization (aolp) and
#     Degree of Linear Polarization (dolp).

#     Args:
#         aolp (np.ndarray): Angle of Linear Polarisation
#         dolp (np.ndarray): Degree of Linear Polarisation

#     Returns:
#         tuple[np.ndarray, np.ndarray, np.ndarray]: normals_diffuse, normals_spec1, normals_spec2 priors.
#     """
#     # refractive index assumed to be ~ 1.5
#     n = 1.5

#     # solve for rho and phi
#     phi = aolp
#     rho = dolp

#     # Calculate diffuse reflection solution
#     # Solve for the angle of incidence (theta)
#     num = (n - 1 / n) ** 2
#     den = 2 * rho * n**2 - rho * (n + 1 / n) ** 2 + 2 * rho
#     sin_theta_diffuse_sq = num / den
#     sin_theta_diffuse = np.sqrt(sin_theta_diffuse_sq)
#     cos_theta_diffuse = np.sqrt(1 - sin_theta_diffuse_sq)
#     theta_diffuse = np.arcsin(sin_theta_diffuse)

#     # Calculate specular reflection solutions
#     # Adjust angle of polarization for specular reflections
#     phi_spec = phi + np.pi / 2

#     # Generate a range of possible sin_theta values
#     sin_theta_spec = np.linspace(-1, 1, 1000)

#     # Calculate corresponding rho values for specular reflections
#     rho_spec = (
#         2
#         * sin_theta_spec**2
#         * np.sqrt(n**2 - sin_theta_spec**2)
#         / (n**2 - sin_theta_spec**2 + 2 * sin_theta_spec**4)
#     )

#     # Interpolate to find angles of incidence for specular reflections
#     theta_spec1, theta_spec2 = np.interp(
#         rho, rho_spec, np.arcsin(sin_theta_spec), left=np.nan, right=np.nan
#     )

#     # Calculate normal vectors for different reflections
#     normals_diffuse = np.stack(
#         [
#             np.cos(phi) * sin_theta_diffuse,
#             np.sin(phi) * sin_theta_diffuse,
#             cos_theta_diffuse,
#         ],
#         axis=-1,
#     )
#     normals_spec1 = np.stack(
#         [
#             np.cos(phi_spec) * np.sin(theta_spec1),
#             np.sin(phi_spec) * np.sin(theta_spec1),
#             np.cos(theta_spec1),
#         ],
#         axis=-1,
#     )
#     normals_spec2 = np.stack(
#         [
#             np.cos(phi_spec) * np.sin(theta_spec2),
#             np.sin(phi_spec) * np.sin(theta_spec2),
#             np.cos(theta_spec2),
#         ],
#         axis=-1,
#     )
#     return normals_diffuse, normals_spec1, normals_spec2
