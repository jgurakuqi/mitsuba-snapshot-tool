from colorama import Fore
import numpy as np
from torch import cuda
import cv2 as cv
import utils
import mitsuba as mi

import pickle
from glob import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
from scipy.io import savemat



def I2channels( I ):
    """Naively demosaic a PFA image to I0, I45, I90, I135. The output images are half the with and height of the input

    Args:
        I (numpy array): Input mosaiced image

    Returns:
        I0, I45, I90, I135: Demosaiced camera channels (half size)
    """    
    assert I.dtype == np.float32 or I.dtype == np.float64
    assert I.ndim == 2


    I90 = I[::2,::2]
    I0 = I[1::2,1::2]
    I45 = I[::2,1::2]
    I135 = I[1::2,::2]

    return I0, I45, I90, I135


def channels2stokes( I0, I45, I90, I135 ):
    """Computes Stokes vector from PFA camera channels

    Args:
        I0 ([type]): Channel I0
        I45 ([type]): Channel I1
        I90 ([type]): Channel I2
        I135 ([type]): Channel I3

    Returns:
        S0,S1,S2: Stokes vector
    """    
    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135
    return S0, S1, S2 


def aolp( S0, S1, S2 ):
    """ Computes Angle of Linear Polarization from Stokes parameters
    """    
    return 0.5 * np.arctan2(S2,S1)


def dolp( S0, S1, S2 ):
    """ Computes Degree of Linear Polarization from Stokes parameters
    """    
    return np.sqrt( S1**2 + S2**2 ) / S0



def compute_priors(image, S0, S1, S2, aolp, dolp):
    with open('deepSfP_priors_reverse.pkl', 'rb') as f:
        interps = pickle.load(f)

    # OUR_DATA_DIR = "./OUR_DATA_ARUCO/"
    OUT_DIR = "./OUT_DATA_OUT/"

    # Check if input folder exists.
    # assert os.path.exists(OUR_DATA_DIR)

    try:
        # Create output folder if not existing.
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
            print(f"Folder '{OUT_DIR}' created successfully.")
        else:
            print(f"Folder '{OUT_DIR}' already exists.")
    except OSError as e:
        print(f"Error creating folder: {e}")

    H = 1024
    W = 1224

    image_files = sorted(glob(f"{OUR_DATA_DIR}/*.png"))

    N = 1 #len(image_files) #300

    # maxi = 20

    for img_id, curr_img_file in enumerate(image_files[0:N]):
        # if img_id == 20:
        #     break

    filename = os.path.basename(curr_img_file)
    print(f"Processing {filename} ({img_id+1}/{N})")
    mask_file = f"{curr_img_file}_mask.npy"

    # load RT
    pose_file = curr_img_file+"_pose.txt"
    if os.path.isfile(pose_file):
        RT = np.loadtxt(pose_file)
        R = RT[:,:3]
    else:
        print(pose_file + "not found, skipping")
        continue
    
    # load mask
    if os.path.isfile(mask_file):
        mask_img = np.load(mask_file).astype(bool)
    else:
        print("mask not found!")
        continue
    
    # normals GT
    normals_gt = np.zeros((H, W, 3))
    normals_gt[mask_img,:] = R[:,2]
    normals_gt[...,1] *= -1
    normals_gt[...,2] *= -1
    
    # plt.figure()
    # plt.imshow(normals_gt[...,0])
    # plt.colorbar()
    
    # load image and demosaic
    img = cv.imread(curr_img_file, cv.IMREAD_GRAYSCALE)
    img = img.astype(float)/255
    I0, I45, I90, I135 = I2channels( img )

    # TODO: read stokes, aolp and dolp
    S0, S1, S2 = channels2stokes( I0, I45, I90, I135 )
    aolp_img = aolp(S0,S1,S2)
    dolp_img = dolp(S0,S1,S2)
    
    W = S0.shape[1]
    H = S0.shape[0]
    
    
    aolp_img_f = aolp_img.flatten()
    dolp_img_f = dolp_img.flatten()
    mask_img_f = mask_img.flatten()

    curr_prior = np.zeros((H*W,9))

    for i in range(9):
        print("interp ", i)
        curr_prior[mask_img_f,i] = interps[i](aolp_img_f[mask_img_f], dolp_img_f[mask_img_f])

    curr_prior = np.reshape(curr_prior, (H, W, 9))

    # plt.figure()
    # plt.imshow(curr_prior[...,0])
    # plt.colorbar()
    
    mat_d = {}
    mat_d["normals_prior"] = curr_prior
    mat_d["mask"] = mask_img.astype(int)
    mat_d["normals_gt"] = normals_gt
    mat_d["images"] = np.stack([I0, I45, I90, I135], axis=2)

    print("saving priors in .mat")
    savemat(f"{OUT_DIR}/{filename}_withpriors.mat", mat_d)
    




def write_output_data(
    I: np.ndarray,
    S0: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    # S3,
    normals: np.ndarray,
    positions: np.ndarray,
    index: int,
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
    utils.plot_rgb_image(I)
    # return

    normals = normals.astype(np.double)
    print(f"[Normals shape]: {normals.shape}")
    print(f"[Normal pos 0,0]: {normals[0,0]}")
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

    # mask = dolp.copy()
    # mask[mask > 0.0] = 255.0
    # utils.plot_rgb_image(np.clip(mask, 0, 255).astype(np.uint8))

    mask = normals.copy()
    mask[mask > 0.] = 255.

    utils.plot_rgb_image(np.clip(mask, 0, 255).astype(np.uint8))

    mask = mask.astype(bool)

    # total_polarized_intensity = np.sqrt(S1^2 + S2^2 + S3^2)

    # unpolarized_intensity = S0 - total_polarized_intensity

    savemat(
        "imgs/data.mat",
        {
            "images": unpolarized_intensity,
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
                # "S3": mi.Bitmap.PixelFormat.Y,
                "nn": mi.Bitmap.PixelFormat.XYZ,
                "pos": mi.Bitmap.PixelFormat.XYZ,
            },
        )

        # Produce the output data based on the extracted layers.
        write_output_data(
            # scene_file_path=scene_file_path,
            I=layers_dict["<root>"],
            S0=layers_dict["S0"],
            S1=layers_dict["S1"],
            S2=layers_dict["S2"],
            # S3=layers_dict["S3"],
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

    chosen_shape = "dragon"  # dragon, thai, armadillo, sphere, cube
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
