"""
This module is used for inference functions.

Software Name : marine-detect
SPDX-FileCopyrightText: Copyright (c) Orange Business Services SA
SPDX-License-Identifier: AGPL-3.0-only.

This software is distributed under the GNU Affero General Public License v3.0,
the text of which is available at https://spdx.org/licenses/AGPL-3.0-only.html <https://spdx.org/licenses/AGPL-3.0-only.html>
or see the "LICENSE" file for more details.

Authors: Eléonore Charles
Software description: Object detection models for identifying species in marine environments.
"""

import os

import cv2
import numpy as np
from PIL import ExifTags, Image
from tqdm import tqdm
from ultralytics import YOLO


def save_combined_image(
    images_input_folder_path: str,
    image_name: str,
    output_folder_pred_images: str,
    combined_results: list,
) -> None:
    """
    Saves the results of multiple detections on an image using specified parameters.

    Args:
        images_input_folder_path (str): Path to the folder containing input images.
        image_name (str): Name of the input image.
        output_folder_pred_images (str): Path to the folder where the combined images will be saved.
        combined_results (list): List of detection results.

    Returns:
        None
    """
    # Open the image and check its orientation
    img_path = os.path.join(images_input_folder_path, image_name)
    original_image = Image.open(img_path)

    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(original_image._getexif().items())

        if exif[orientation] == 3:
            original_image = original_image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            original_image = original_image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            original_image = original_image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    output_path = os.path.join(output_folder_pred_images, image_name)
    combined_image = combine_results(np.array(original_image), combined_results)
    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    combined_image = combined_image[..., ::-1]
    combined_image = Image.fromarray(combined_image)
    combined_image.save(output_path)


def combine_results(original_image: np.ndarray, results_list: list) -> np.ndarray:
    """
    Combines results from a list of detection outcomes.

    It uses the original image and returns the resulting combined image array.

    Args:
        original_image (np.ndarray): Array representing the original image.
        results_list (list): List of detection results.

    Returns:
        np.ndarray: Combined image array.
    """
    combined_image = original_image

    for results in results_list:
        for result in results:
            combined_image = result.plot(img=combined_image)

    return combined_image


def predict_on_images(
    model_paths: list[str],
    confs_threshold: list[float],
    images_input_folder_path: str,
    images_output_folder_path: str,
    save_txt: bool = False,
    save_conf: bool = False,
) -> None:
    """
    Utilizes a list of YOLO models to predict detections on a set of images.

    Args:
        model_paths (list[str]): List of paths to YOLO model files.
        confs_threshold (list[float]): List of confidence thresholds corresponding to each model.
        images_input_folder_path (str): Path to the folder containing input images.
        images_output_folder_path (str): Path to the folder where annotated images will be saved.
        save_txt (bool): Whether to save bounding box coordinates in text files.
        save_conf (bool): Whether to save confidence scores in text files.

    Returns:
        None
    """
    models = [YOLO(model_path) for model_path in model_paths]

    if images_output_folder_path:
        os.makedirs(f"{images_output_folder_path}", exist_ok=True)

    for image_name in tqdm(os.listdir(images_input_folder_path)):
        combined_results = []
        for i, model in enumerate(models):
            results = model(
                os.path.join(images_input_folder_path, image_name),
                conf=confs_threshold[i],
                save_txt=save_txt,
                save_conf=save_conf,
            )
            combined_results.extend(results)

        if images_output_folder_path:
            save_combined_image(
                images_input_folder_path,
                image_name,
                images_output_folder_path,
                combined_results,
            )


def predict_on_video(
    model_paths: list[str],
    confs_threshold: list[float],
    input_video_path: str,
    output_video_path: str,
) -> None:
    """
    Processes a video by utilizing a list of YOLO models to predict detections on its frames.

    The annotated frames are then written to an output video file, with parameters such as
    confidence thresholds and input/output file paths specified. The video processing loop continues
    until the end of the input video or until the user presses 'q'.

    Args:
        model_paths (list[str]): List of paths to YOLO model files.
        confs_threshold (list[float]): List of confidence thresholds corresponding to each model.
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to the output video file.

    Returns:
        None
    """
    models = [YOLO(model_path) for model_path in model_paths]
    cap = cv2.VideoCapture(input_video_path)

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Get video frame dimensions and frame rate
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            combined_results = []
            for i, model in enumerate(models):
                results = model(
                    frame,
                    conf=confs_threshold[i],
                )
                combined_results.extend(results)

            # Visualize the results on the frame
            annotated_frame = combine_results(frame, combined_results)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object, VideoWriter, and close any remaining windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
