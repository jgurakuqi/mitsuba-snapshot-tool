from colorama import Fore
import numpy as np
import cv2 as cv
import utils
import mitsuba as mi
import numpy as np
from scipy.io import savemat
import threading
from os.path import exists


def generate_priors(
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
    Computes the priors required by Deep Shape's Neural network and stores normals and foreground
    mask for comparisons among different Shape From Polarization techniques.

    Args:
        S0 (np.ndarray): Stoke 0
        S1 (np.ndarray): Stoke 1
        S2 (np.ndarray): Stoke 2
        aolp (np.ndarray): Angle of linear polarization.
        dolp (np.ndarray): Degree of linear polarization.
        mask (np.ndarray): Foreground mask.
        normals (np.ndarray): Ground truth surface normals.
        fov (float): Camera fov.
        output_directory (str): Path of the output parent folder which contains
        every kind of output. Defaults to "outputs/".
        deep_shape_folder_name (str): Name of the Deep Shape's data folder. Defaults to "for_deep_shape/".
        chosen_shape (str): Chosen shape.
        chosen_camera (str): Chosen camera type (orthographic or perspective).
        chosen_material (str): Chosen shape's material.
        chosen_reflectance (str): Chosen reflectance amount.
    """
    interps = utils.load_interpolations("deepSfP_priors_reverse.pkl")

    # *** Store mask and normals as npy files ***

    output_path = f"{output_directory}{deep_shape_folder_name}{chosen_shape}/"

    # npy_path = f"{output_path}{chosen_shape}_{chosen_camera}_{chosen_material}_{chosen_reflectance}_{fov}_deg_fov_"

    # if not exists(f"{npy_path}MASK.npy"):
    #     np.save(f"{npy_path}MASK", mask)
    # else:
    #     print(f"Following file already exists: {npy_path}MASK.npy")

    # if not exists(f"{npy_path}NORMALS.npy"):
    #     np.save(f"{npy_path}NORMALS", normals)
    # else:
    #     print(f"Following file already exists: {npy_path}NORMALS.npy")

    # *** Check if priors already exist before of computing them ***

    mat_path = f"{output_path}{chosen_shape}_{chosen_camera}_{chosen_material}_{chosen_reflectance}_{fov}_deg_fov.mat"

    if exists(mat_path):
        print("Priors already existing!")
        return

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

    # *** Compute priors with multithreading ***

    curr_prior = np.zeros((height * width, 9))

    def interpolate_thread(i):
        print(f"     - Interpolating {i} ...")
        curr_prior[flattened_mask, i] = interps[i](
            masked_flattened_aolp, masked_flattened_dolp
        )

    threads = [threading.Thread(target=interpolate_thread, args=(i,)) for i in range(9)]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    curr_prior = np.reshape(curr_prior, (height, width, 9))

    # *** Store output data ***

    print("saving priors in .mat")

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
    invoke_generate_priors: bool = True,
    invoke_generate_matlab_data: bool = True,
) -> None:
    """Use the extracted layers from the rendered scene to compute and store information
    required by Deep Shape's Neural network and Smith's Matlab comparison to perform Shape
    From Polarization.

    Args:
        S0 (np.ndarray): Stoke 0
        S1 (np.ndarray): Stoke 1
        S2 (np.ndarray): Stoke 2
        S3 (np.ndarray): Stoke 3
        normals (np.ndarray): Ground truth normals
        specular_amount (float): Amount of specular reflection of the rendered shape.
        fov (float): Chosen Fov for rendered scene.
        chosen_shape (str): Chosen shape.
        chosen_camera (str): Chosen camera type (orthographic or perspective).
        chosen_material (str): Chosen shape's material.
        chosen_reflectance (str): Chosen reflectance amount.
        output_directory (str, optional): Path of the output parent folder which contains
        every kind of output. Defaults to "outputs/".
        comparator_folder_name (str, optional): Name of the comparator output folder for storing
        .mat files. Defaults to "for_comparator/".
        images_folder_name (str, optional): Name of the output/debug images' folder. Defaults to "images/".
        deep_shape_folder_name (str, optional): Name of the Deep Shape's data folder. Defaults to "for_deep_shape/".
        invoke_generate_priors (bool, optional): Tells whether to perform priors' computation
        or not. Defaults to True.
        invoke_generate_matlab_data (bool, optional): Tells whether to compute or not the mat data required
        by Smith's matlab comparator.
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

    # *** Pack and store the polarization data for Matlab comparator. ***

    mat_path = f"{output_directory}{comparator_folder_name}{chosen_shape}/"
    mat_path = f"{mat_path}{chosen_shape}_{chosen_camera}_{chosen_material}_{chosen_reflectance}_{fov}_deg_fov.mat"

    if not exists(mat_path) and invoke_generate_matlab_data:
        spec_mask = (
            mask.astype(bool) if specular_amount != 0.0 else (mask * 0).astype(bool)
        )
        savemat(
            f"{output_directory}{comparator_folder_name}{chosen_shape}/{chosen_shape}_{chosen_camera}_{chosen_material}_{chosen_reflectance}_{fov}_deg_fov.mat",
            {
                # "images": S0,
                "unpol": S0 - np.sqrt(S1**2 + S2**2 + S3**2),
                "dolp": dolp,
                "aolp": aolp,
                "mask": mask.astype(bool),
                "spec": spec_mask,
                # "gt_normals": normals,
            },
        )
    else:
        print(f"Following file already exists: {mat_path}")

    # *** Compute input (with priors) for Deep Shape's Neural network. ***

    if invoke_generate_priors:
        generate_priors(
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
    sample_count: int,
    scenes_folder_path: str = "./scene_files/",
    camera_width: int = 1224,
    camera_height: int = 1024,
    invoke_generate_priors: bool = True,
    invoke_generate_matlab_data: bool = True,
) -> None:
    """Capture data from a scene using Mitsuba 3 and store the data of the rendering
    for the Deep Shape's Neural Network and for the Smith's matlab comparator.

    Args:
        chosen_shape (str): Chosen shape.
        chosen_camera (str): Chosen camera type (orthographic or perspective).
        chosen_material (str): Chosen shape's material.
        chosen_reflectance (str): Chosen reflectance amount.
        fov (float): Chosen fov.
        fov_scale (float): Scale factor to re-scale the chosen shape according to the
        fov.
        scenes_folder_path (str, optional): Path to the scene file to render.. Defaults to "./scene_files/".
        camera_width (int, optional): Rendered scene's width. Defaults to 1224.
        camera_height (int, optional): Rendered scene's height. Defaults to 1024.
        sample_count (int, optional): Number of samples per pixel. Defaults to 16.
        invoke_generate_priors (bool, optional): Tells whether the priors for Deep Shape must be
        computed or not. Defaults to True.
        invoke_generate_matlab_data (bool, optional): Tells whether to compute or not the mat data required
        by Smith's matlab comparator.

    Raises:
        ValueError: Exception thrown if the rendering fails.
    """
    reflectance_types = {
        # "specular": {"specular": 1.0, "diffuse": 0.0},
        "diffuse": {"diffuse": 1.0, "specular": 0.0},
        "specular": {"diffuse": 0.2, "specular": 1.0},
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
            invoke_generate_priors=invoke_generate_priors,
            invoke_generate_matlab_data=invoke_generate_matlab_data,
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
    chosen_reflectance = "specular"  # diffuse, specular, _
    chosen_sample_count = 1  # 1, 16, 56, 90, 156, 256

    fovs_dict_keys = fovs_dict[chosen_shape].keys()
    fovs_dict_len = len(fovs_dict_keys)

    invoke_generate_priors = False  # ! FALSE PREVENTS PRIORS

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
            invoke_generate_priors=invoke_generate_priors,
        )

        print(f"Processed {current_fov_index + 1} of {fovs_dict_len}")


def main_all() -> None:
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
    chosen_material = "pplastic"  # pplastic, conductor
    chosen_reflectance = "realistic_specular"  # diffuse, specular, realistic_specular
    chosen_sample_count = 132  # 1, 16, 56, 90, 156, 256

    invoke_generate_priors = False  # ! -------------------------------------------
    invoke_generate_matlab_data = False

    scale_factor = 0.3
    fovs_dict["cube"] = {
        key: (value * scale_factor) if key != 0 else "Orthografic"
        for (key, value) in fovs_dict["sphere"].items()
    }
    scale_factor = 1.3
    fovs_dict["dragon"] = {
        key: (value * scale_factor) if key != 0 else "Orthografic"
        for (key, value) in fovs_dict["sphere"].items()
    }

    scale_factor = 2.15
    fovs_dict["bunny"] = {
        key: (value * scale_factor) if key != 0 else "Orthografic"
        for (key, value) in fovs_dict["sphere"].items()
    }

    scale_factor = 2.7
    fovs_dict["armadillo"] = {
        key: (value * scale_factor) if key != 0 else "Orthografic"
        for (key, value) in fovs_dict["sphere"].items()
    }

    scale_factor = 1.9
    fovs_dict["pyramid"] = {
        key: (value * scale_factor) if key != 0 else "Orthografic"
        for (key, value) in fovs_dict["sphere"].items()
    }

    shapes_dict_keys = fovs_dict.keys()
    shapes_dict_len = len(shapes_dict_keys)
    for current_shape_index, current_shape in enumerate(fovs_dict.keys()):
        fovs_dict_keys = fovs_dict[current_shape].keys()
        fovs_dict_len = len(fovs_dict_keys)
        for current_fov_index, current_fov in enumerate(
            fovs_dict[current_shape].keys()
        ):
            print(
                f"[{current_shape}]: Start processing of {current_fov_index + 1}/{fovs_dict_len}..."
            )
            capture_scene(
                fov=current_fov,
                fov_scale=fovs_dict[current_shape][current_fov],
                sample_count=chosen_sample_count,
                chosen_shape=current_shape,
                chosen_camera="orth" if current_fov == 0 else "persp",
                chosen_material=chosen_material,
                chosen_reflectance=chosen_reflectance,
                invoke_generate_priors=invoke_generate_priors,
                invoke_generate_matlab_data=invoke_generate_matlab_data,
            )

            print(
                f"[{current_shape}]: Processed {current_fov_index + 1}/{fovs_dict_len}"
            )
        print(f"Completed {current_shape_index + 1} shapes out of {shapes_dict_len}")


if __name__ == "__main__":
    main_all()
