from colorama import Fore
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
import torch
import cv2 as cv

import xml.etree.ElementTree as ET


def free_gpu_memory() -> None:
    """
    Free the GPU's main memory. Should be called after scripts (e.g., Mitsuba's load_file)
    that don't free memory on their own.
    """
    torch.cuda.empty_cache()


mi.set_variant("cuda_mono_polarized")


def extract_layer_as_numpy(
    film: mi.Film, name: str, pxformat: mi.Bitmap.PixelFormat
) -> np.ndarray:
    """
    Extract a layer from the film as a NumPy array.

    Args:
        film (mitsuba.Film): The film object from which to extract the layer.
        name (str): Name of the layer to extract.
        pxformat (mitsuba.Bitmap.PixelFormat): Pixel format for the extracted layer.

    Returns:
        numpy.ndarray: The extracted layer as a NumPy array.
    """

    for layer in film.bitmap(raw=False).split():
        if layer[0] == name:
            return np.array(
                layer[1].convert(pxformat, mi.Struct.Type.Float32, srgb_gamma=False)
            )
    raise ValueError(f"Layer -- {name} -- not found")


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
    camera_width: int = 1280,
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
        print(f"{Fore.LIGHTMAGENTA_EX} Exception: {e}")
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

    # plot_rgb_image(I)

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


# def xml_appender(path)
#     # Parse the XML strings
#     root_1 = ET.fromstring(xml_file_1)
#     root_2 = ET.fromstring(xml_file_2)

#     # Append the first XML content as a child of the <scene> tag in the second XML
#     root_2.append(root_1)

#     # Convert the updated XML element tree back to a string
#     updated_xml = ET.tostring(root_2, encoding='unicode')

#     print(updated_xml)


def insert_substring(original_string, substring, index):
    return original_string[:index] + substring + original_string[index:]


def insert_xml_object():
    # Read content of "ciccio1.xml"
    with open("scene_no_sphere.xml", "r") as file1:
        scene_no_sphere = file1.read()

    with open("sphere_conductor.xml", "r") as file2:
        sphere_conductor = file2.read()

    scene_no_sphere = insert_substring(
        scene_no_sphere,
        sphere_conductor,
        -9,
    )
    # print(f"scene_no_sphere: {scene_no_sphere}")

    with open("scene.xml", "w") as file1:
        file1.write(scene_no_sphere)


def main() -> None:
    insert_xml_object()
    # return

    # camera_width = 1024
    # camera_height = 768
    DEBUG_STOP_ITERATION = 1
    camera_width = 1920
    camera_height = 1080
    sample_count = 256
    scene = "scene.xml"

    total = len(np.linspace(0, 360, 60))
    print("Start processing:\n")
    for angle_index, current_angle in enumerate(np.linspace(0, 360, 60)):
        if DEBUG_STOP_ITERATION == angle_index:
            print(f"[DEBUG]: PROCESSING STOPPED AT ITERATION {DEBUG_STOP_ITERATION}")
            return
        print(f"Starting with angle {angle_index + 1}/{total}...")
        capture_scene(
            scene,
            index=angle_index,
            camera_width=camera_width,
            camera_height=camera_height,
            angle=current_angle,
            sample_count=sample_count,  # Increase the sample count for better quality
        )
        torch.cuda.empty_cache()
        print(f"{angle_index + 1}/{total} processed.\n")


if __name__ == "__main__":
    main()
