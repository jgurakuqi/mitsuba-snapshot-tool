# --------------------------------------------------------------------------------------------------------------
# ! THIS file contains some utilities which allow to create dinamically scenes from a recipient scene and a simple mitsuba abstraction
# ! defined into another file.
# ! ALSO, the commented functions are the foundations of a script for allowing the creation of complex scenes given several elements, i.e.,
# ! a recipient scene and several abstraction files.

# --------------------------------------------------------------------------------------------------------------


# def xml_abstraction_insert(
#     identifier: str,
#     sub_file_path: str,
#     final_file_path: str,
#     recipient_file_content: str = None,
#     recipient_file_path: str = None,
#     perform_indentation: bool = True,
#     indentation_spaces: int = 4,
# ) -> list[str]:
#     """
#     Insert the XML content of the file located in sub_file_path into the file located in recipient_file_path,
#     specifically inside of the tag indetified through the given identifier, which can be any text. If the used
#     identifier is not unique, the first element located will be selected as insertion point.
#     It's also possible to choose whether the new XML file should be correctly indented, and if so also the
#     number of spaces used by the running system for indentation.

#     Args:
#         recipient_file_path (str): Path to the file whose content will be extended.
#         identifier (str): XML content to locate in the recipient file.
#         sub_file_path (str): Path to the file used to extend the recipient.
#         final_file_path (str): Path to the file which will contain the extended content.
#         perform_indentation (bool): Tells whether to perform indentation or not. Defaults to True.
#         indentation_spaces (int, optional): Number of spaces used in case of indentation. Defaults to 4.

#     Raises:
#         ValueError: Exception thrown if the required id is not found.
#     """
#     # The recipient_file_content will be initialized with the content of the file is the var is empty.
#     if recipient_file_content == None:
#         if recipient_file_path == None:
#             raise ValueError(
#                 "[XML builder]: Recipient file content and path can't be both empty."
#             )
#         with open(sub_file_path, "r") as abstraction_file:
#             recipient_file_content = abstraction_file.readlines()

#     with open(recipient_file_path, "r") as main_file:
#         lines = main_file.readlines()

#     found_line = None
#     white_spaces_number = None
#     # Find the matching abstraction through its id.
#     for i, line in enumerate(lines):
#         if identifier in line:
#             # print(f"Found line: {i}")
#             if perform_indentation:
#                 white_spaces_number = (
#                     len(lines[i]) - len(lines[i].lstrip()) + indentation_spaces
#                 )
#             found_line = i
#             break

#     # If found, insert the new sub-abstraction, otherwise raise exception.
#     if found_line is not None:
#         if perform_indentation:
#             # Indent using the found number of spaces.
#             white_spaces = " " * white_spaces_number
#             recipient_file_content = [
#                 white_spaces + line for line in recipient_file_content
#             ]

#         # Concatenate file contents as lists.
#         lines = (
#             lines[: found_line + 1]
#             + recipient_file_content
#             + ["\n"]
#             + lines[found_line + 1 :]
#         )
#         # print(f"lines: {lines}")
#         # with open(final_file_path, "w") as final_file:
#         #     final_file.writelines(lines)
#         return lines
#     else:
#         raise ValueError(f"[insert_content_after_id]: LINE not found.")


# def build_scene(scene_path: str, abstractions_filenames: list[str]):
#     temp_result = None
#     for abstraction_filename in abstractions_filenames:
#         temp_result = xml_abstraction_insert(
#             recipient_file_path=scene_path,
#             identifier="scene",
#             sub_file_path=abstraction_filename,
#         )
#     with open("".join(), "w") as final_file:
#         final_file.writelines(lines)


def xml_abstraction_insert(
    recipient_file_path: str,
    identifier: str,
    sub_file_path: str,
    final_file_path: str,
    perform_indentation: bool = True,
    indentation_spaces: int = 4,
) -> list[str]:
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
        # with open(final_file_path, "w") as final_file:
        #     final_file.writelines(lines)
        return lines
    else:
        raise ValueError(f"[insert_content_after_id]: LINE not found.")

    # recipient_file_name = "scene.xml"
    # sub_file_name = (
    #     "sphere_pplastic.xml"  # ["sphere_conductor.xml", "sphere_conductor.xml"]
    # )
    # final_scene_name = recipient_file_name.replace(".xml", f"_{sub_file_name}")

    # # Create scene xml with required objects.
    # xml_abstraction_insert(
    #     recipient_file_path=scene_files_path + recipient_file_name,
    #     identifier="scene",
    #     sub_file_path=scene_files_path + sub_file_name,
    #     final_file_path=scene_files_path + final_scene_name,
    # )
