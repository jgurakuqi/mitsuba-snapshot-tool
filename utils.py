import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt

# (llvm, cuda) + (mono, spectral) + (polarized)
mi.set_variant("llvm_mono_polarized")


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
                mi.Struct.Type.Float32,
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
