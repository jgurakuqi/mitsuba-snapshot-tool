from colorama import Fore
import numpy as np
from torch import cuda
import cv2 as cv
import utils
import mitsuba as mi
from scipy.io import savemat


def calc_priors(
    aolp: np.ndarray, dolp: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the physical priors based on Angle of Linear Polarization (aolp) and
    Degree of Linear Polarization (dolp).

    Args:
        aolp (np.ndarray): Angle of Linear Polarisation
        dolp (np.ndarray): Degree of Linear Polarisation

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: normals_diffuse, normals_spec1, normals_spec2 priors.
    """
    n = 1.5  # refractive index

    # solve for rho and phi
    phi = aolp
    rho = dolp

    # Calculate diffuse reflection solution
    # Solve for the angle of incidence (theta)
    num = (n - 1 / n) ** 2
    den = 2 * rho * n**2 - rho * (n + 1 / n) ** 2 + 2 * rho
    sin_theta_diffuse_sq = num / den
    sin_theta_diffuse = np.sqrt(sin_theta_diffuse_sq)
    cos_theta_diffuse = np.sqrt(1 - sin_theta_diffuse_sq)
    theta_diffuse = np.arcsin(sin_theta_diffuse)

    # Calculate specular reflection solutions
    # Adjust angle of polarization for specular reflections
    phi_spec = phi + np.pi / 2

    # Generate a range of possible sin_theta values
    sin_theta_spec = np.linspace(-1, 1, 1000)

    # Calculate corresponding rho values for specular reflections
    rho_spec = (
        2
        * sin_theta_spec**2
        * np.sqrt(n**2 - sin_theta_spec**2)
        / (n**2 - sin_theta_spec**2 + 2 * sin_theta_spec**4)
    )

    # Interpolate to find angles of incidence for specular reflections
    theta_spec1, theta_spec2 = np.interp(
        rho, rho_spec, np.arcsin(sin_theta_spec), left=np.nan, right=np.nan
    )

    # Calculate normal vectors for different reflections
    normals_diffuse = np.stack(
        [
            np.cos(phi) * sin_theta_diffuse,
            np.sin(phi) * sin_theta_diffuse,
            cos_theta_diffuse,
        ],
        axis=-1,
    )
    normals_spec1 = np.stack(
        [
            np.cos(phi_spec) * np.sin(theta_spec1),
            np.sin(phi_spec) * np.sin(theta_spec1),
            np.cos(theta_spec1),
        ],
        axis=-1,
    )
    normals_spec2 = np.stack(
        [
            np.cos(phi_spec) * np.sin(theta_spec2),
            np.sin(phi_spec) * np.sin(theta_spec2),
            np.cos(theta_spec2),
        ],
        axis=-1,
    )
    return normals_diffuse, normals_spec1, normals_spec2


def capture_scene(
    scene_file_path: str,
    index: int,
    camera_width: int = 1024,
    camera_height: int = 720,
    angle: float = 0.0,
    sample_count: int = 16,
    tilt: float = 0.0,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Capture data from a scene using Mitsuba.

    Args:
        scene_file_path (str): Path to the scene file to render.
        index (int): index of the current snapshot angle, used to define the path.
        camera_width (int, optional): Width of the camera sensor. Defaults to 1024.
        camera_height (int, optional): Height of the camera sensor. Defaults to 768.
        angle (float, optional): Camera rotation angle in degrees. Defaults to 0.
        sample_count (int, optional): Number of samples to use for rendering. Defaults to 3.
        tilt (float, optional): Camera tilt angle in degrees. Defaults to 0.

    Returns:
        tuple: A tuple containing the captured data as NumPy arrays:
            - I: RGB image (camera rendering)
            - S0, S1, S2, S3: Stokes parameters
            - normals: Surface normals
            - positions: Surface positions
    """
    try:
        scene = mi.load_file(
            scene_file_path,
            camera_width=camera_width,
            camera_height=camera_height,
            angle=angle,
            sample_count=sample_count,
            tilt=tilt,
        )
    except Exception as e:
        print(f"{Fore.LIGHTMAGENTA_EX} Exception: {e}.{Fore.WHITE}")
        return

    sensor = scene.sensors()[0]
    scene.integrator().render(scene, sensor)
    film = sensor.film()

    I = utils.extract_layer_as_numpy(film, "<root>", mi.Bitmap.PixelFormat.RGB)
    S0 = utils.extract_layer_as_numpy(film, "S0", mi.Bitmap.PixelFormat.Y)
    S1 = utils.extract_layer_as_numpy(film, "S1", mi.Bitmap.PixelFormat.Y)
    S2 = utils.extract_layer_as_numpy(film, "S2", mi.Bitmap.PixelFormat.Y)
    normals = utils.extract_layer_as_numpy(film, "nn", mi.Bitmap.PixelFormat.XYZ)
    positions = utils.extract_layer_as_numpy(film, "pos", mi.Bitmap.PixelFormat.XYZ)

    utils.plot_rgb_image(I)
    return

    normals = normals.astype(np.double)
    positions = positions.astype(np.double)

    # ! Added to prevent Zero-Divisions in Dolp computation.
    S0[S0 == 0] = np.finfo(float).eps

    aolp = 0.5 * np.arctan2(S2, S1)
    dolp = np.sqrt(S1**2 + S2**2) / S0
    # dolp[S0==0] = 0

    cv.imwrite(f"imgs/I_{index}.png", np.clip(I * 255, 0, 255).astype(np.uint8))
    cv.imwrite(f"imgs/S0_{index}.png", np.clip(S0 * 255, 0, 255).astype(np.uint8))
    cv.imwrite(f"imgs/DOLP_{index}.png", (dolp * 255).astype(np.uint8))

    # # print(S0.shape, type(S0))
    # # S0_gray = cv.cvtColor(S0, cv.COLOR_RGB2GRAY)
    # S0_scaled = np.clip(S0 * 255, 0, 255).astype(np.uint8)
    # spec = cv.threshold(S0_scaled, 250, 1, cv.THRESH_BINARY)[1]

    # binarized_logic = np.ndarray(S0_scaled.shape, dtype=bool)
    # spec_logic = np.ndarray(S0_scaled.shape, dtype=bool)

    # binarized_logic[::] = False
    # binarized_logic[
    #     cv.threshold(S0_scaled, 0, 1, cv.THRESH_OTSU + cv.THRESH_BINARY)[1] == 1
    # ] = True

    # spec_logic[::] = False
    # spec_logic[spec == 1] = True

    # savemat(
    #     "imgs/data.mat",
    #     {
    #         "S0": S0,
    #         "dolp": dolp,
    #         "aolp": aolp,
    #         "mask": binarized_logic,
    #         "spec": spec_logic,
    #     },
    # )

    angle_n = cv.applyColorMap(
        ((aolp + np.pi / 2) / np.pi * 255.0).astype(np.uint8), cv.COLORMAP_HSV
    )
    cv.imwrite(f"imgs/AOLP_{index}.png", angle_n)

    cv.imwrite(f"imgs/N_{index}.png", ((normals + 1.0) * 0.5 * 255).astype(np.uint8))

    # np.savez(f"imgs/stokes_{index}.npz", S0=S0, S1=S1, S2=S2, dolp=dolp, aolp=aolp)


def main() -> None:
    debug_stop_iteration = 1
    # delete_scene_file = False
    camera_width = 1920
    camera_height = 1450
    sample_count = 16  # Higher means better quality - 16, 156, 256
    scene_files_path = "./scene_files/"

    chosen_shape = "cube"  # dragon, thai, armadillo, sphere, cube
    chosen_camera = "orth"  # orth, persp
    chosen_material = "pplastic"  # pplastic, conductor
    polarization_type = ""  # , ext_lens

    if polarization_type != "":
        polarization_type = f"_{polarization_type}"

    scene_path = f"{scene_files_path}{chosen_shape}/{chosen_camera}_{chosen_material}{polarization_type}.xml"

    total = len(range(0, 360, 60))
    total = total if total < debug_stop_iteration else debug_stop_iteration
    print("Start processing:\n")

    # cuda.init()

    # Start capturing the scene from different angles:
    for angle_index, current_angle in enumerate(range(0, 360, 60)):
        cuda.empty_cache()
        if debug_stop_iteration == angle_index:
            # In case of DEBUG-testing, stops the execution at the required iteration.
            print(f"[DEBUG]: PROCESSING STOPPED AT ITERATION {debug_stop_iteration}")
            return
        print(f"Starting with angle {angle_index + 1}/{total}...")
        capture_scene(
            scene_file_path=scene_path,
            index=angle_index,
            camera_width=camera_width,
            camera_height=camera_height,
            angle=current_angle,
            sample_count=sample_count,
        )
        print(f"{angle_index + 1}/{total} processed.\n")


if __name__ == "__main__":
    main()
