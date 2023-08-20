from colorama import Fore
import numpy as np
from torch import cuda
import cv2 as cv
import utils
import mitsuba as mi
from scipy.io import savemat


def write_output_data(
    scene_file_path: str,
    I: np.ndarray,
    S0: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    normals: np.ndarray,
    positions: np.ndarray,
    index: int,
) -> None:
    """
    Use the extracted layers from the rendered scene to compute and store information such
    as the Degree and Angle of linear polarization, normals and other data.

    Args:
        scene_file_path (str): Path to the scene file to render.
        I (np.ndarray): Intensities.
        S0 (np.ndarray): Stoke 0
        S1 (np.ndarray): Stoke 1
        S2 (np.ndarray): Stoke 2
        normals (np.ndarray): _description_
        positions (np.ndarray): _description_
        index (int): index of the current snapshot angle, used to define the path.
    """
    utils.plot_rgb_image(I)
    # return

    normals = normals.astype(np.double)
    positions = positions.astype(np.double)

    # ! Added to prevent Zero-Divisions in Dolp computation.
    S0[S0 == 0] = np.finfo(float).eps

    aolp = 0.5 * np.arctan2(S2, S1)
    dolp = np.sqrt(S1**2 + S2**2) / S0
    # dolp[S0==0] = 0

    angle_n = cv.applyColorMap(
        ((aolp + np.pi / 2) / np.pi * 255.0).astype(np.uint8), cv.COLORMAP_HSV
    )

    cv.imwrite(f"imgs/I_{index}.png", np.clip(I * 255, 0, 255).astype(np.uint8))
    cv.imwrite(f"imgs/S0_{index}.png", np.clip(S0 * 255, 0, 255).astype(np.uint8))
    cv.imwrite(f"imgs/DOLP_{index}.png", (dolp * 255).astype(np.uint8))
    cv.imwrite(f"imgs/AOLP_{index}.png", angle_n)
    cv.imwrite(f"imgs/N_{index}.png", ((normals + 1.0) * 0.5 * 255).astype(np.uint8))

    # S0_scaled = np.clip(dolp * 255.0, 0, 255).astype(np.uint8)
    # if scene_file_path.find("conductor"):
    #     # For conductors:
    #     mask = cv.threshold(S0_scaled, 1, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)[1]
    # else:
    #     # For pplastics:
    #     mask = cv.threshold(S0_scaled, 1, 255, cv.THRESH_BINARY)[1]

    mask = dolp.copy()
    mask[mask > 0.0] = 255.0

    utils.plot_rgb_image(np.clip(mask, 0, 255).astype(np.uint8))

    mask = mask.astype(bool)

    savemat(
        "imgs/data.mat",
        {
            "images": S0,
            "dolp": dolp,
            "aolp": aolp,
            "mask": mask,
            "spec": mask,
        },
    )


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
        # Capture the scene.
        scene = mi.load_file(
            scene_file_path,
            camera_width=camera_width,
            camera_height=camera_height,
            angle=angle,
            sample_count=sample_count,
            tilt=tilt,
        )

        sensor = scene.sensors()[0]
        scene.integrator().render(scene, sensor)
        film = sensor.film()

        # Extract the film's layers (i.e., stokes, normals, etc...).
        layers_dict = utils.extract_chosen_layers_as_numpy(
            film,
            {
                "<root>": mi.Bitmap.PixelFormat.RGB,
                "S0": mi.Bitmap.PixelFormat.Y,
                "S1": mi.Bitmap.PixelFormat.Y,
                "S2": mi.Bitmap.PixelFormat.Y,
                "nn": mi.Bitmap.PixelFormat.XYZ,
                "pos": mi.Bitmap.PixelFormat.XYZ,
            },
        )

        # Produce the output data based on the extracted layers.
        write_output_data(
            scene_file_path=scene_file_path,
            I=layers_dict["<root>"],
            S0=layers_dict["S0"],
            S1=layers_dict["S1"],
            S2=layers_dict["S2"],
            normals=layers_dict["nn"],
            positions=layers_dict["pos"],
            index=index,
        )
    except Exception as e:
        print(f"{Fore.LIGHTMAGENTA_EX} Exception: {e}.{Fore.WHITE}")
        return


def main() -> None:
    debug_stop_iteration = 1
    # delete_scene_file = False
    camera_width = 1920
    camera_height = 1450
    sample_count = 16  # Higher means better quality - 16, 156, 256
    scene_files_path = "./scene_files/"

    chosen_shape = "sphere"  # dragon, thai, armadillo, sphere, cube
    chosen_camera = "orth"  # orth, persp
    chosen_material = "conductor"  # pplastic, conductor
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
