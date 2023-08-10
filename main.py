from colorama import Fore
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

# from torch import cuda
import cv2 as cv

# import os

# llvm_mono_polarized, cuda_mono_polarized, scalar_mono_polarized
mi.set_variant("llvm_mono_polarized")


def extract_layer_as_numpy(
    film: mi.Film, name: str, pxformat: mi.Bitmap.PixelFormat
) -> np.ndarray:
    """
    Extract a layer from the film as a NumPy array.

    Args:
        film (mitsuba.Film): The film object from which to extract the layer.
        name (str): Name of the layer to extract.
        pxformat (mitsuba.Bitmap.PixelFormat): Pixel format for the extracted layer.

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


def plot_rgb_image(image: np.ndarray) -> None:
    """
    Plot the RGB image.

    Args:
        image (numpy.ndarray): The RGB image as a 2D NumPy array.
    """
    plt.imshow(image)
    plt.axis("off")
    plt.show()


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

    I = extract_layer_as_numpy(film, "<root>", mi.Bitmap.PixelFormat.RGB)
    S0 = extract_layer_as_numpy(film, "S0", mi.Bitmap.PixelFormat.Y)
    S1 = extract_layer_as_numpy(film, "S1", mi.Bitmap.PixelFormat.Y)
    S2 = extract_layer_as_numpy(film, "S2", mi.Bitmap.PixelFormat.Y)
    # S3 = extract_layer_as_numpy(film, "S3", mi.Bitmap.PixelFormat.Y)
    normals = extract_layer_as_numpy(film, "nn", mi.Bitmap.PixelFormat.XYZ)
    positions = extract_layer_as_numpy(film, "pos", mi.Bitmap.PixelFormat.XYZ)

    plot_rgb_image(I)
    # return

    normals = normals.astype(np.double)
    positions = positions.astype(np.double)

    aolp = 0.5 * np.arctan2(S2, S1)
    dolp = np.sqrt(S1**2 + S2**2) / S0
    # dolp[S0==0] = 0

    cv.imwrite(f"imgs/I_{index}.png", np.clip(I * 255, 0, 255).astype(np.uint8))
    cv.imwrite(f"imgs/S0_{index}.png", np.clip(S0 * 255, 0, 255).astype(np.uint8))
    cv.imwrite(f"imgs/DOLP_{index}.png", (dolp * 255).astype(np.uint8))

    angle_n = cv.applyColorMap(
        ((aolp + np.pi / 2) / np.pi * 255.0).astype(np.uint8), cv.COLORMAP_HSV
    )
    cv.imwrite(f"imgs/AOLP_{index}.png", angle_n)

    cv.imwrite(f"imgs/N_{index}.png", ((normals + 1.0) * 0.5 * 255).astype(np.uint8))

    np.savez(f"imgs/stokes_{index}.npz", S0=S0, S1=S1, S2=S2, dolp=dolp, aolp=aolp)
    # print(f"I: {I}, S0: {S0}, S1: {S1}, S2: {S2}, S3: {S3}\n\n")
    # return I, S0, S1, S2, S3, normals, positions


def main() -> None:
    debug_stop_iteration = 1
    # delete_scene_file = False
    camera_width = 1920
    camera_height = 1080
    sample_count = 16  # Higher means better quality - 256
    scene_files_path = "./scene_files/"

    # ! perspective camera scenes
    # persp_pplastic_dragon, persp_pplastic_armadillo, persp_pplastic_sphere, persp_pplastic_cube, persp_conductor_cube, persp_conductor_sphere

    # ! orthographic camera scenes
    # _, _, orth_pplastic_sphere, orth_pplastic_cube, orth_conductor_cube, orth_conductor_sphere

    scene_path = f"{scene_files_path}orth_conductor_sphere.xml"

    total = len(range(0, 360, 60))
    print("Start processing:\n")

    # cuda.init()

    # Start capturing the scene from different angles:
    for angle_index, current_angle in enumerate(range(0, 360, 60)):
        # cuda.empty_cache()
        if debug_stop_iteration == angle_index:
            # In case of DEBUG-testing, stops the execution at the required iteration.
            print(f"[DEBUG]: PROCESSING STOPPED AT ITERATION {debug_stop_iteration}")
            return
        print(f"Starting with angle {angle_index + 1}/{total}...")
        capture_scene(
            # scene_file_path = scene_files_path + final_scene_name,
            scene_file_path=scene_path,
            index=angle_index,
            camera_width=camera_width,
            camera_height=camera_height,
            angle=current_angle,
            sample_count=sample_count,
        )
        print(f"{angle_index + 1}/{total} processed.\n")

    # if delete_scene_file:
    #     del_path = scene_files_path + final_scene_name
    #     try:
    #         os.remove(del_path)
    #         print(f"File '{del_path}' has been successfully deleted.")
    #     except FileNotFoundError:
    #         print(f"File '{del_path}' not found.")
    #     except PermissionError:
    #         print(f"Permission denied. Unable to delete '{del_path}'.")
    #     except Exception as e:
    #         print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
