from colorama import Fore
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
import torch
import cv2 as cv
import os


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


def xml_abstraction_insert(
    recipient_file_path: str,
    identifier: str,
    sub_file_path: str,
    final_file_path: str,
    perform_indentation: bool = True,
    indentation_spaces: int = 4,
):
    """
    Insert the XML content of the file located in sub_file_path into the file located in recipient_file_path,
    specifically inside of the tag indetified through the given identifier, which can be any text. If the used
    identifier is not unique, the first element located will be selected as insertion point.
    It's also possible to choose whether the new XML file should be correctly indented, and if so also the
    number of spaces used by the running system for indentation.

    Args:
        recipient_file_path (str): Path to the file whose content will be extended.
        identifier (str): XML content to locate in the recipient file.
        sub_file_path (str): Path to the file used to extend the recipient.
        final_file_path (str): Path to the file which will contain the extended content.
        perform_indentation (bool): Tells whether to perform indentation or not. Defaults to True.
        indentation_spaces (int, optional): Number of spaces used in case of indentation. Defaults to 4.

    Raises:
        ValueError: Exception thrown if the required id is not found.
    """
    # Open sub and recipient files as list of strings.
    with open(sub_file_path, "r") as abstraction_file:
        abstraction_file_content = abstraction_file.readlines()
    with open(recipient_file_path, "r") as main_file:
        lines = main_file.readlines()

    found_line = None
    white_spaces_number = None
    # Find the matching abstraction through its id.
    for i, line in enumerate(lines):
        if identifier in line:
            # print(f"Found line: {i}")
            if perform_indentation:
                white_spaces_number = (
                    len(lines[i]) - len(lines[i].lstrip()) + indentation_spaces
                )
            found_line = i
            break

    # If found, insert the new sub-abstraction, otherwise raise exception.
    if found_line is not None:
        if perform_indentation:
            # Indent using the found number of spaces.
            white_spaces = " " * white_spaces_number
            abstraction_file_content = [
                white_spaces + line for line in abstraction_file_content
            ]

        # Concatenate file contents as lists.
        lines = (
            lines[: found_line + 1]
            + abstraction_file_content
            + ["\n"]
            + lines[found_line + 1 :]
        )
        # print(f"lines: {lines}")
        with open(final_file_path, "w") as final_file:
            final_file.writelines(lines)
    else:
        raise ValueError(f"[insert_content_after_id]: LINE not found.")


def main() -> None:
    debug_stop_iteration = 1
    delete_scene_file = False
    camera_width = 1920
    camera_height = 1080
    sample_count = 16  # Higher means better quality - 256
    scene_files_path = "./scene_files/"
    recipient_file_name = "scene.xml"
    sub_file_name = (
        "sphere_pplastic.xml"  # ["sphere_conductor.xml", "sphere_conductor.xml"]
    )
    final_scene_name = recipient_file_name.replace(".xml", f"_{sub_file_name}")

    # Create scene xml with required objects.
    xml_abstraction_insert(
        recipient_file_path=scene_files_path + recipient_file_name,
        identifier="scene",
        sub_file_path=scene_files_path + sub_file_name,
        final_file_path=scene_files_path + final_scene_name,
    )

    total = len(np.linspace(0, 360, 60))
    print("Start processing:\n")

    # Start capturing the scene from different angles:
    for angle_index, current_angle in enumerate(np.linspace(0, 360, 60)):
        if debug_stop_iteration == angle_index:
            # In case of DEBUG-testing, stops the execution at the required iteration.
            print(f"[DEBUG]: PROCESSING STOPPED AT ITERATION {debug_stop_iteration}")
            return
        print(f"Starting with angle {angle_index + 1}/{total}...")
        capture_scene(
            scene_files_path + final_scene_name,
            index=angle_index,
            camera_width=camera_width,
            camera_height=camera_height,
            angle=current_angle,
            sample_count=sample_count,
        )
        torch.cuda.empty_cache()
        print(f"{angle_index + 1}/{total} processed.\n")

    if delete_scene_file:
        del_path = scene_files_path + final_scene_name
        try:
            os.remove(del_path)
            print(f"File '{del_path}' has been successfully deleted.")
        except FileNotFoundError:
            print(f"File '{del_path}' not found.")
        except PermissionError:
            print(f"Permission denied. Unable to delete '{del_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
