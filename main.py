from colorama import Fore
import numpy as np
from torch import cuda
import cv2 as cv
import utils
import mitsuba as mi
import pickle
import numpy as np
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


def compute_priors(
    S0: np.ndarray,
    aolp: np.ndarray,
    dolp: np.ndarray,
    mask: np.ndarray,
    normals: np.ndarray,
    angle: float,
    output_directory: str,
    deep_shape_folder_name: str,
    chosen_shape: str,
    chosen_camera: str,
    chosen_material: str,
    chosen_reflectance: str,
) -> None:
    """
    Computes the priors required by Deep Shape's Neural network.

    Args:
        S0 (np.ndarray): stoke 0, i.e., total intensity.
        aolp (np.ndarray): Angle of linear polarization.
        dolp (np.ndarray): Degree of linear polarization.
        mask (np.ndarray): Binary mask for targeting only the prominent object.
        normals (np.ndarray): Surface normals of said object.
        output_directory (str): directory path for storing outputs.
    """
    interps = load_interpolations("deepSfP_priors_reverse.pkl")

    # ground truth normals.
    normals[:, 1] *= -1
    normals[:, 2] *= -1

    height, width = S0.shape[:2]

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

    curr_prior = np.zeros((height * width, 9))

    def interpolate_thread(i):
        print(f"Interpolating {i} ...")
        curr_prior[flattened_mask, i] = interps[i](
            masked_flattened_aolp, masked_flattened_dolp
        )

    threads = [threading.Thread(target=interpolate_thread, args=(i,)) for i in range(9)]
    for thread in threads:
        thread.start()

    # widthait for all threads to finish
    for thread in threads:
        thread.join()

    curr_prior = np.reshape(curr_prior, (height, width, 9))

    print("saving priors in .mat")

    output_path = f"{output_directory}{deep_shape_folder_name}{chosen_shape}/"

    savemat(
        f"{output_path}{chosen_shape}_{chosen_camera}_{chosen_material}_{chosen_reflectance}_{angle}.mat",
        {
            "normals_prior": curr_prior,
            "mask": mask.astype(int),
            "normals_gt": normals,
            "images": S0,  # np.stack([I0, I45, I90, I135], axis=2)
        },
    )


def write_output_data(
    S0: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    S3: np.ndarray,
    normals: np.ndarray,
    specular_amount: float,
    angle: float,
    chosen_shape: str,
    chosen_camera: str,
    chosen_material: str,
    chosen_reflectance: str,
    output_directory: str = "outputs/",
    comparator_folder_name: str = "for_comparator/",
    images_folder_name: str = "images/",
    deep_shape_folder_name: str = "for_deep_shape/",
    invoke_compute_priors: bool = True,
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
    """
    # *** Compute polarization info ***

    S0[S0 == 0] = np.finfo(float).eps  # Prevent Zero-Divisions in Dolp computation.
    # np.set_printoptions(threshold=np.inf)
    # print(f"S2: {S2}\n\n\nS1: {S1}")
    aolp = 0.5 * np.arctan2(S2, S1)
    # print(f"\n\n\nAolp: {aolp}\n\n")
    # return
    dolp = np.sqrt(S1**2 + S2**2) / S0
    angle_n = cv.applyColorMap(
        ((aolp + np.pi / 2) / np.pi * 255.0).astype(np.uint8), cv.COLORMAP_HSV
    )

    # *** Check if all the output folders exist. If not, create the missing ones. ***

    utils.check_output_folders(
        chosen_shape=chosen_shape,
        output_directory=output_directory,
        comparator_folder_name=comparator_folder_name,
        images_folder_name=images_folder_name,
        deep_shape_folder_name=deep_shape_folder_name,
    )

    # *** Store the output images ***
    # ! Chosen reflectance might be temporary
    utils.write_output_images(
        S0=S0,
        dolp=dolp,
        angle_n=angle_n,
        normals=normals,
        output_directory=output_directory,
        images_folder_name=images_folder_name,
        chosen_shape=chosen_shape,
        chosen_camera=chosen_camera,
        chosen_material=chosen_material,
        chosen_reflectance=chosen_reflectance,
        angle=angle,
    )

    # if "conductor" in output_name:
    #     # spec_mask = mask * 0  spec_mask[S0 > 5.0] = 100 spec_mask = mask.astype(bool)
    #     # Alternative:
    #     # temp_S0 = np.clip(S0 * 255.0, 0, 255).astype(np.uint8) temp_S0[temp_S0 == 255] = 0
    #     # temp_S0[temp_S0 < 230] = 0  spec_mask = temp_S0.astype(bool)
    #     spec_mask = S0.astype(bool)
    # else: spec_mask = (mask.astype(bool) if specular_amount != 0.0 else (mask * 0).astype(bool))

    # *** Compute masks for Deep Shape Neural Network and Matlab comparator. ***

    mask = (np.sum(np.square(normals), axis=-1) > 0.0).astype(np.uint8)
    # utils.plot_rgb_image(np.clip(mask, 0, 255).astype(np.uint8))
    spec_mask = mask.astype(bool) if specular_amount != 0.0 else (mask * 0).astype(bool)

    savemat(
        f"{output_directory}{comparator_folder_name}{chosen_shape}/{chosen_shape}_{chosen_camera}_{chosen_material}_{chosen_reflectance}_{angle}.mat",
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

    if invoke_compute_priors:
        compute_priors(
            S0=S0,
            aolp=aolp,
            dolp=dolp,
            mask=mask,
            normals=normals,
            angle=angle,
            output_directory=output_directory,
            deep_shape_folder_name=deep_shape_folder_name,
            chosen_shape=chosen_shape,
            chosen_camera=chosen_camera,
            chosen_material=chosen_material,
            chosen_reflectance=chosen_reflectance,
        )


def capture_scene(
    camera_width: int,
    camera_height: int,
    chosen_shape: str,
    chosen_camera: str,
    chosen_material: str,
    chosen_reflectance: str,
    scenes_folder_path: str,
    angle: float,
    sample_count: int = 16,
    invoke_compute_priors: bool = True,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Capture data from a scene using Mitsuba.

    Args:
        scene_file_path (str): Path to the scene file to render.
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

    scene_file_path = (
        f"{scenes_folder_path}{chosen_shape}/{chosen_camera}_{chosen_material}.xml"
    )
    try:
        # ----- Load the scene ----.
        if "conductor" in scene_file_path:
            # Use default reflectance coefficient with conductors.
            scene = mi.load_file(
                scene_file_path,
                camera_width=camera_width,
                camera_height=camera_height,
                angle=angle,
                sample_count=sample_count,
            )
        else:
            # Use chosen coefficients with pplastics.
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
            S0=layers_dict["S0"],
            S1=layers_dict["S1"],
            S2=layers_dict["S2"],
            S3=layers_dict["S3"],
            normals=layers_dict["nn"],
            angle=angle,
            specular_amount=specular_amount,
            chosen_shape=chosen_shape,
            chosen_camera=chosen_camera,
            chosen_material=chosen_material,
            chosen_reflectance=chosen_reflectance,
            invoke_compute_priors=invoke_compute_priors,
        )
    except Exception as e:
        # print(f"{Fore.LIGHTMAGENTA_EX} Exception: {e}.{Fore.WHITE}")
        raise ValueError(f"{Fore.LIGHTMAGENTA_EX} Exception: {e}.{Fore.WHITE}")
        # return


def main() -> None:
    debug_stop_iteration = 1

    total = len(range(0, 360, 60))
    total = total if total < debug_stop_iteration else debug_stop_iteration
    print("Start processing:\n")

    # ? cuda.init()
    # Start capturing the scene from different angles:
    for angle_index, current_angle in enumerate(range(0, 360, 60)):
        # ? cuda.empty_cache()
        if debug_stop_iteration == angle_index:
            print(f"[DEBUG]: PROCESSING STOPPED AT ITERATION {debug_stop_iteration}")
            return
        print(f"Starting with angle {angle_index + 1}/{total}...")
        capture_scene(
            camera_width=1224,
            camera_height=1024,
            angle=current_angle,
            sample_count=56,  # Higher --> better quality (16, 156, 256)
            chosen_shape="cube",  # dragon, thai, armadillo, bunny, sphere, cube, pyramid, plane
            chosen_camera="orth",  # orth, persp
            chosen_material="pplastic",  # pplastic, conductor
            chosen_reflectance="diffuse",  # diffuse, specular, mixed, realistic
            scenes_folder_path="./scene_files/",
            invoke_compute_priors=False,  # ! FALSE PREVENTS PRIORS COMPUTATION
        )
        print(f"{angle_index + 1}/{total} processed.\n")


if __name__ == "__main__":
    main()
