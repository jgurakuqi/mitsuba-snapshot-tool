from colorama import Fore
import numpy as np
import cv2 as cv
import utils
import mitsuba as mi
import numpy as np
from scipy.io import savemat
import threading


def compute_priors(
    S0: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    aolp: np.ndarray,
    dolp: np.ndarray,
    mask: np.ndarray,
    normals: np.ndarray,
    fov: float,
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
    interps = utils.load_interpolations("deepSfP_priors_reverse.pkl")

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
        print(f"     - Interpolating {i} ...")
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

    I0, I45, I90, I135 = utils.simulate_pfa_mosaic(S0, S1, S2)

    savemat(
        f"{output_path}{chosen_shape}_{chosen_camera}_{chosen_material}_{chosen_reflectance}_{fov}_deg_fov.mat",
        {
            "normals_prior": curr_prior,
            "mask": mask.astype(int),
            "normals_gt": normals,  # np.clip(normals),
            "images": np.stack([I0, I45, I90, I135], axis=2),  # S0
        },
    )


def write_output_data(
    S0: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    S3: np.ndarray,
    normals: np.ndarray,
    specular_amount: float,
    fov: float,
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
    np.set_printoptions(threshold=np.inf)
    aolp = 0.5 * np.arctan2(S2, S1)
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
        fov=fov,
    )
    # *** Compute masks for Deep Shape Neural Network and Matlab comparator. ***

    mask = (np.sum(np.square(normals), axis=-1) > 0.0).astype(np.uint8)
    spec_mask = mask.astype(bool) if specular_amount != 0.0 else (mask * 0).astype(bool)

    savemat(
        f"{output_directory}{comparator_folder_name}{chosen_shape}/{chosen_shape}_{chosen_camera}_{chosen_material}_{chosen_reflectance}_{fov}_deg_fov.mat",
        {
            # "images": S0,
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
            S1=S1,
            S2=S2,
            aolp=aolp,
            dolp=dolp,
            mask=mask,
            normals=normals,
            fov=fov,
            output_directory=output_directory,
            deep_shape_folder_name=deep_shape_folder_name,
            chosen_shape=chosen_shape,
            chosen_camera=chosen_camera,
            chosen_material=chosen_material,
            chosen_reflectance=chosen_reflectance,
        )


def capture_scene(
    chosen_shape: str,
    chosen_camera: str,
    chosen_material: str,
    chosen_reflectance: str,
    fov: float,
    fov_scale: float,
    scenes_folder_path: str = "./scene_files/",
    camera_width: int = 1224,
    camera_height: int = 1024,
    sample_count: int = 16,
    invoke_compute_priors: bool = True,
) -> None:
    """
    Capture data from a scene using Mitsuba.

    Args:
        scene_file_path (str): Path to the scene file to render.
        camera_width (int, optional): Width of the camera sensor. Defaults to 1024.
        camera_height (int, optional): Height of the camera sensor. Defaults to 768.
        fov (float):
        sample_count (int, optional): Number of samples to use for rendering. Defaults to 3.
    """
    reflectance_types = {
        "specular": {"specular": 1.0, "diffuse": 0.0},
        "diffuse": {"diffuse": 1.0, "specular": 0.0},
        "realistic_specular": {"diffuse": 0.1, "specular": 1.0},
    }
    specular_amount = reflectance_types[chosen_reflectance]["specular"]
    diffuse_amount = reflectance_types[chosen_reflectance]["diffuse"]

    scene_file_path = (
        f"{scenes_folder_path}{chosen_shape}/{chosen_camera}_{chosen_material}.xml"
    )
    try:
        if "orth" in scene_file_path:
            # Orthografic camera
            scene = mi.load_file(
                scene_file_path,
                camera_width=camera_width,
                camera_height=camera_height,
                sample_count=sample_count,
                diffuse=diffuse_amount,
                specular=specular_amount,
            )
        else:
            if fov_scale != None:
                # Perspective camera with re-scaling.
                scene = mi.load_file(
                    scene_file_path,
                    camera_width=camera_width,
                    camera_height=camera_height,
                    sample_count=sample_count,
                    diffuse=diffuse_amount,
                    specular=specular_amount,
                    fov=fov,
                    computed_scale=fov_scale,
                )
            else:
                # Perspective camera without re-scaling.
                scene = mi.load_file(
                    scene_file_path,
                    camera_width=camera_width,
                    camera_height=camera_height,
                    sample_count=sample_count,
                    diffuse=diffuse_amount,
                    specular=specular_amount,
                    fov=fov,
                )

        sensor = scene.sensors()[0]
        scene.integrator().render(scene, sensor)

        # *** Extract film's layers (i.e., stokes, normals, etc...) ***

        layers_dict = utils.extract_chosen_layers_as_numpy(
            sensor.film(),
            {
                "S0": mi.Bitmap.PixelFormat.Y,
                "S1": mi.Bitmap.PixelFormat.Y,
                "S2": mi.Bitmap.PixelFormat.Y,
                "S3": mi.Bitmap.PixelFormat.Y,
                "nn": mi.Bitmap.PixelFormat.XYZ,
            },
        )

        # *** Produce the output data based on the extracted layers. ***

        write_output_data(
            S0=layers_dict["S0"],
            S1=layers_dict["S1"],
            S2=layers_dict["S2"],
            S3=layers_dict["S3"],
            normals=layers_dict["nn"],
            specular_amount=specular_amount,
            chosen_shape=chosen_shape,
            chosen_camera=chosen_camera,
            chosen_material=chosen_material,
            chosen_reflectance=chosen_reflectance,
            invoke_compute_priors=invoke_compute_priors,
            fov=fov,
        )
    except Exception as e:
        raise ValueError(f"{Fore.LIGHTMAGENTA_EX} Exception: {e}.{Fore.WHITE}")


def main() -> None:
    fovs_dict = {
        "sphere": {
            10: 0.25,
            20: 0.5,
            30: 0.75,
            40: 1.0,
            50: 1.25,
            60: 1.5,
            70: 1.75,
            80: 2.0,
            90: 2.25,
            0: "Orthografic",
        },
    }
    chosen_shape = "plane"  # dragon, armadillo, bunny, sphere, cube, pyramid, plane
    chosen_material = "pplastic"  # pplastic, conductor
    chosen_reflectance = "diffuse"  # diffuse, specular, realistic_specular
    chosen_sample_count = 1  # 1, 16, 56, 90, 156, 256

    fovs_dict_keys = fovs_dict[chosen_shape].keys()
    fovs_dict_len = len(fovs_dict_keys)

    invoke_compute_priors = False  # ! FALSE PREVENTS PRIORS

    match chosen_shape:
        case "cube":
            scale_factor = 0.3

            fovs_dict["cube"] = {
                key: (value * scale_factor) if key != 0 else "Orthografic"
                for (key, value) in fovs_dict["sphere"].items()
            }
        case "dragon":
            scale_factor = 1.3

            fovs_dict["dragon"] = {
                key: (value * scale_factor) if key != 0 else "Orthografic"
                for (key, value) in fovs_dict["sphere"].items()
            }

        case "bunny":
            scale_factor = 2.15

            fovs_dict["bunny"] = {
                key: (value * scale_factor) if key != 0 else "Orthografic"
                for (key, value) in fovs_dict["sphere"].items()
            }

        case "armadillo":
            scale_factor = 2.7

            fovs_dict["armadillo"] = {
                key: (value * scale_factor) if key != 0 else "Orthografic"
                for (key, value) in fovs_dict["sphere"].items()
            }

        case "pyramid":
            scale_factor = 1.9

            fovs_dict["pyramid"] = {
                key: (value * scale_factor) if key != 0 else "Orthografic"
                for (key, value) in fovs_dict["sphere"].items()
            }
        case _:
            raise ValueError("The chosen shape does not exist!")

    for current_fov_index, current_fov in enumerate(fovs_dict[chosen_shape].keys()):
        # if current_fov == 0:
        capture_scene(
            fov=current_fov,
            fov_scale=fovs_dict[chosen_shape][current_fov],
            sample_count=chosen_sample_count,
            chosen_shape=chosen_shape,
            chosen_camera="orth" if current_fov == 0 else "persp",
            chosen_material=chosen_material,
            chosen_reflectance=chosen_reflectance,
            invoke_compute_priors=invoke_compute_priors,
        )

        print(f"Processed {current_fov_index + 1} of {fovs_dict_len}")


if __name__ == "__main__":
    main()
