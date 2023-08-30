from colorama import Fore
import numpy as np
from torch import cuda
import cv2 as cv
import utils
import mitsuba as mi
import pickle
import numpy as np

# import matplotlib.pyplot as plt
import os
from scipy.io import savemat

import threading


def load_interpolations(file_path: str):
    """
    Load the interpolators.

    Args:
        file_path (str): path to the interpolators.

    Returns:
        _type_: _description_
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
        if not os.path.exists(directory):
            print(f"Folder '{directory}' does not exists. Starting creation...")
            os.makedirs(directory)
            print(f"Folder '{directory}' created successfully.")
    except OSError as e:
        print(f"Error creating folder: {e}")


def compute_priors(
    S0: np.ndarray,
    aolp: np.ndarray,
    dolp: np.ndarray,
    mask: np.ndarray,
    normals: np.ndarray,
    index: int,
    output_directory: str,
    output_name: str,
) -> None:
    """
    Computes the priors required by Deep Shape's Neural network.

    Args:
        S0 (np.ndarray): stoke 0, i.e., total intensity.
        aolp (np.ndarray): Angle of linear polarization.
        dolp (np.ndarray): Degree of linear polarization.
        mask (np.ndarray): Binary mask for targeting only the prominent object.
        normals (np.ndarray): Surface normals of said object.
        index (int): index of current object's frame.
        output_directory (str): directory path for storing outputs.
    """
    interps = load_interpolations("deepSfP_priors_reverse.pkl")

    output_directory += "for_deep_shape/"

    # Create output folder if not existing.
    create_directory(output_directory)

    # ground truth normals.
    normals[:, 1] *= -1
    normals[:, 2] *= -1

    H, W = S0.shape[:2]

    flattened_aolp, flattened_dolp, flattened_mask = (
        aolp.flatten(),
        dolp.flatten(),
        mask.flatten(),
    )

    masked_flattened_aolp, masked_flattened_dolp = (
        flattened_aolp[flattened_mask],
        flattened_dolp[flattened_mask],
    )

    # ----- Compute priors -----

    curr_prior = np.zeros((H * W, 9))

    def interpolate_thread(i):
        print(f"Interpolating {i} ...")
        curr_prior[flattened_mask, i] = interps[i](
            masked_flattened_aolp, masked_flattened_dolp
        )

    threads = [threading.Thread(target=interpolate_thread, args=(i,)) for i in range(9)]
    for thread in threads:
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    curr_prior = np.reshape(curr_prior, (H, W, 9))

    print("saving priors in .mat")

    savemat(
        f"{output_directory}{output_name}_{index}_with_priors.mat",
        {
            "normals_prior": curr_prior,
            "mask": mask.astype(int),
            "normals_gt": normals,
            "images": S0,  # np.stack([I0, I45, I90, I135], axis=2)
        },
    )


def write_output_data(
    output_name: str,
    # I: np.ndarray,
    S0: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    S3: np.ndarray,
    normals: np.ndarray,
    specular_amount: float,
    # positions: np.ndarray,
    index: int,
    output_directory: str = "outputs/",
    comparator_folder_name: str = "for_comparator/",
) -> None:
    """
    Use the extracted layers from the rendered scene to compute and store information such
    as the Degree and Angle of linear polarization, normals and other data.

    Args:
        I (np.ndarray): Intensities.
        S0 (np.ndarray): Stoke 0
        S1 (np.ndarray): Stoke 1
        S2 (np.ndarray): Stoke 2
        normals (np.ndarray): _description_
        positions (np.ndarray): _description_
        index (int): index of the current snapshot angle, used to define the path.
    """
    S0[S0 == 0] = np.finfo(float).eps  # Prevent Zero-Divisions in Dolp computation.
    aolp = 0.5 * np.arctan2(S2, S1)
    dolp = np.sqrt(S1**2 + S2**2) / S0
    angle_n = cv.applyColorMap(
        ((aolp + np.pi / 2) / np.pi * 255.0).astype(np.uint8), cv.COLORMAP_HSV
    )

    # ----- Write computed data as files. -----

    imgs_path = f"{output_directory}{output_name}_{index}_"
    # cv.imwrite(f"{imgs_path}I.png", np.clip(I * 255.0, 0, 255).astype(np.uint8))
    cv.imwrite(f"{imgs_path}S0.png", np.clip(S0 * 255.0, 0, 255).astype(np.uint8))
    cv.imwrite(f"{imgs_path}DOLP.png", (dolp * 255.0).astype(np.uint8))
    cv.imwrite(f"{imgs_path}AOLP_COLOURED.png", angle_n)
    cv.imwrite(
        f"{imgs_path}NORMALS.png", ((normals + 1.0) * 0.5 * 255.0).astype(np.uint8)
    )

    # ----- Compute binary mask. -----

    # mask = normals.copy()
    # mask[mask > 0.0] = 255.0
    mask = (np.sum(np.square(normals), axis=-1) > 0.0).astype(np.uint8)

    # utils.plot_rgb_image(np.clip(mask, 0, 255).astype(np.uint8))

    comparator_folder_path = f"{output_directory}{comparator_folder_name}"

    # Create comparator output folder if not existing.
    create_directory(comparator_folder_path)

    # ----- Store .mat data for William's Matlab comparator. -----

    spec_mask = mask.astype(bool) if specular_amount != 0.0 else (mask * 0).astype(bool)

    print(f"Spec values: {set(spec_mask.flatten())}")

    savemat(
        f"{comparator_folder_path}{output_name}_{index}.mat",
        {
            "images": S0,
            "unpol": S0 - np.sqrt(S1**2 + S2**2 + S3**2),
            "dolp": dolp,
            "aolp": aolp,
            "mask": mask.astype(bool),
            "spec": spec_mask,
        },
    )

    # ----- Compute input (with priors) for Deep Shape's Neural network. -----

    # compute_priors(S0, aolp, dolp, mask, normals, index, output_directory, output_name)


def capture_scene(
    output_name: str,
    scene_file_path: str,
    index: int,
    chosen_reflectance: str,
    camera_width: int,
    camera_height: int,
    angle: float = 0.0,
    sample_count: int = 16,
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
    reflectance_types = {
        "specular": {"specular": 1.0, "diffuse": 0.0},
        "diffuse": {"diffuse": 1.0, "specular": 0.0},
        "mixed": {"diffuse": 0.5, "specular": 0.5},
        "realistic": {"diffuse": 0.2, "specular": 1.0},
    }
    specular_amount = reflectance_types[chosen_reflectance]["specular"]
    diffuse_amount = reflectance_types[chosen_reflectance]["diffuse"]
    try:
        # ----- Load the scene ----.
        scene = mi.load_file(
            scene_file_path,
            camera_width=camera_width,
            camera_height=camera_height,
            angle=angle,
            sample_count=sample_count,
            diffuse=diffuse_amount,
            specular=specular_amount,
        )

        sensor = scene.sensors()[0]
        scene.integrator().render(scene, sensor)

        # ----- Extract film's layers (i.e., stokes, normals, etc...) -----

        layers_dict = utils.extract_chosen_layers_as_numpy(
            sensor.film(),
            {
                # "<root>": mi.Bitmap.PixelFormat.RGB,
                "S0": mi.Bitmap.PixelFormat.Y,
                "S1": mi.Bitmap.PixelFormat.Y,
                "S2": mi.Bitmap.PixelFormat.Y,
                "S3": mi.Bitmap.PixelFormat.Y,
                "nn": mi.Bitmap.PixelFormat.XYZ,
                "pos": mi.Bitmap.PixelFormat.XYZ,
            },
        )

        # ----- Produce the output data based on the extracted layers. -----

        write_output_data(
            output_name=output_name,
            # I=layers_dict["<root>"],
            S0=layers_dict["S0"],
            S1=layers_dict["S1"],
            S2=layers_dict["S2"],
            S3=layers_dict["S3"],
            normals=layers_dict["nn"],
            # positions=layers_dict["pos"],
            index=index,
            specular_amount=specular_amount,
        )
    except Exception as e:
        print(f"{Fore.LIGHTMAGENTA_EX} Exception: {e}.{Fore.WHITE}")
        return


def main() -> None:
    debug_stop_iteration = 1
    camera_width = 1224  # 1920
    camera_height = 1024  # 1450
    sample_count = 56  # Higher means better quality - 16, 156, 256
    scene_files_path = "./scene_files/"

    chosen_shape = "dragon"  # dragon, thai, armadillo, sphere, cube
    chosen_camera = "persp"  # orth, persp
    chosen_material = "pplastic"  # pplastic, conductor
    chosen_reflectance = "diffuse"  # diffuse, specular, mixed, realistic

    # * OK sphere_persp_pplastic_diffuse
    # * OK dragon_persp_pplastic_diffuse

    scene_path = (
        f"{scene_files_path}{chosen_shape}/{chosen_camera}_{chosen_material}.xml"
    )

    total = len(range(0, 360, 60))
    total = total if total < debug_stop_iteration else debug_stop_iteration
    print("Start processing:\n")

    # ? cuda.init()

    # Start capturing the scene from different angles:
    for angle_index, current_angle in enumerate(range(0, 360, 60)):
        # ? cuda.empty_cache()
        if debug_stop_iteration == angle_index:
            # In case of DEBUG-testing, stops the execution at the required iteration.
            print(f"[DEBUG]: PROCESSING STOPPED AT ITERATION {debug_stop_iteration}")
            return
        print(f"Starting with angle {angle_index + 1}/{total}...")
        capture_scene(
            output_name=f"{chosen_shape}_{chosen_camera}_{chosen_material}_{chosen_reflectance}",
            scene_file_path=scene_path,
            index=angle_index,
            camera_width=camera_width,
            camera_height=camera_height,
            angle=current_angle,
            sample_count=sample_count,
            chosen_reflectance=chosen_reflectance,
        )
        print(f"{angle_index + 1}/{total} processed.\n")


if __name__ == "__main__":
    main()
