# Marine Detect 🌊🐟

This repository provides access to two **YOLOv8 object detection models** for identifying **species** of interest in **underwater environments**.

<p align="center">
<img src="./assets/gif/results.gif" height="250"/>
<img src="./assets/gif/turtle+fish-resized.gif" height="250"/>
</p>

These models were developed in the context of the [Let's Revive](https://letsrevive.orange.com/technology-partner/) project in partnership with [Tēnaka](https://tenaka.org/). Tēnaka emphasizes impact measurement through Tēnaka Science, sharing monthly coral ecosystem data. To automate data collection, Orange Business and Tēnaka partner to deploy an underwater research station with cameras. The system uses AI to identify and quantify 17 species in coral areas, streamlining monitoring and allowing marine biologists more time for restoration.

Images: @copyright Tēnaka

## 🐟 Species Scope

The *Fish and Invertebrates* Object Detection Model detects the *Fish and Invertebrates Species* and the *MegaFauna* Object Detection Model detects *MegaFauna and Rare Species*.

- **MegaFauna and Rare Species**: Sharks, Sea Turtles, Rays.
- **Fish Species**: Butterfly Fish (Chaetodontidae), Grouper (Serranidae), Parrotfish (Scaridae), Snapper (Lutjanidae), Moray Eel (Muraenidae), Sweet Lips (Haemulidae), Barramundi Cod (Cromileptes altivelis), Humphead (Napoleon) Wrasse (Cheilinus undulatus), Bumphead Parrotfish (Bolbometopon muricatum), Fish (other than above or unrecognizable).
- **Invertebrates Species**: Giant Clam, Urchin, Sea Cucumber, Lobster, Crown of Thorns.

These species are **"bio-indicating"** species, which serve as indicators of the ecosystem health. These bio-indicating species are of course dependent on each region - here the focus is for Malaysia/Indo-Pacific region.

## 📊 Datasets Details

The models utilize a combination of publicly available datasets (~ 90%) and Tēnaka-based datasets (~ 10%). Some datasets were already annotated, and others undergo manual labeling.

References to the public datasets used can be found in the 'References' section of this README.

The images used with annotations (YOLO format) can be downloaded using the following links: [FishInv dataset](https://stpubtenakanclyw.blob.core.windows.net/marine-detect/FishInv-dataset.zip?sp=r&st=2024-02-16T12:11:04Z&se=2025-02-16T20:11:04Z&spr=https&sv=2022-11-02&sr=c&sig=p68C1yNzj49DEgoJZQCX3uG5MK%2FQ3J%2BwIBE9l7NrEGw%3D), [MegaFauna dataset](https://stpubtenakanclyw.blob.core.windows.net/marine-detect/MegaFauna-dataset.zip?sp=r&st=2024-02-16T12:11:04Z&se=2025-02-16T20:11:04Z&spr=https&sv=2022-11-02&sr=c&sig=p68C1yNzj49DEgoJZQCX3uG5MK%2FQ3J%2BwIBE9l7NrEGw%3D).

### Datasets split details

| Model          | Training + Validation Sets | Test Set     |
| -------------- | -------------------------- | --------     |
| FishInv        | 12,243 images (80%, 20%)   | 499  images  |
| MegaFauna      | 8,130 images (80%, 20%)    | 253  images  |

> [!NOTE]
> The rationale behind the development of two distinct models lies in the utilization of already annotated images available in public datasets. By having separate models, we sidestep the necessity of reannotating images that already encompass annotations for specific species with every Fish, Invertebrates and MegaFauna species.  For example, we found a lot of images of turtles already annotated. If we were to adopt a single, all-encompassing model for both Fish and Invertebrates Species 🐟 and MegaFauna 🦈, it would necessitate the reannotation of all those turtle images to include species like urchins, fishes, ...

## 🤖 Model Details

You can download the trained models using the following links: [FishInv model](https://stpubtenakanclyw.blob.core.windows.net/marine-detect/models/FishInv.pt?sp=r&st=2024-02-16T12:11:04Z&se=2025-02-16T20:11:04Z&spr=https&sv=2022-11-02&sr=c&sig=p68C1yNzj49DEgoJZQCX3uG5MK%2FQ3J%2BwIBE9l7NrEGw%3D), [MegaFauna model](https://stpubtenakanclyw.blob.core.windows.net/marine-detect/models/MegaFauna.pt?sp=r&st=2024-02-16T12:11:04Z&se=2025-02-16T20:11:04Z&spr=https&sv=2022-11-02&sr=c&sig=p68C1yNzj49DEgoJZQCX3uG5MK%2FQ3J%2BwIBE9l7NrEGw%3D).

### Performances on test sets

> [!IMPORTANT]
> Our models are currently undergoing enhancements for improved performance. More labeled images are on the way and will be used to retrain the models.

<details>

<summary>MegaFauna model performances</summary>

| Class  | Images | Instances | mAP50 | mAP50-95 |
| ------ | ------ | --------- | ----- | -------- |
| ray    | 253    | 73        | 0.863 | 0.777    |
| shark  | 253    | 111       | 0.741 | 0.627    |
| turtle | 253    | 109       | 0.948 | 0.887    |

</details>

<details>

<summary>FishInv model performances</summary>

| Class                  | Images | Instances | mAP50 | mAP50-95 |
| ---------------------- | ------ | --------- | ----- | -------- |
| fish                   | 499    | 259       | 0.616 | 0.501    |
| serranidae             | 499    | 49        | 0.850 | 0.777    |
| urchin                 | 499    | 80        | 0.743 | 0.479    |
| scaridae               | 499    | 48        | 0.828 | 0.794    |
| chaetodontidae         | 499    | 65        | 0.891 | 0.827    |
| giant_clam             | 499    | 102       | 0.870 | 0.602    |
| lutjanidae             | 499    | 86        | 0.865 | 0.777    |
| muraenidae             | 499    | 58        | 0.949 | 0.809    |
| sea_cucumber           | 499    | 33        | 0.969 | 0.939    |
| haemulidae             | 499    | 22        | 0.972 | 0.945    |
| lobster                | 499    | 31        | 0.984 | 0.877    |
| crown_of_thorns        | 499    | 28        | 0.981 | 0.790    |
| bolbometopon_muricatum | 499    | 19        | 0.993 | 0.936    |
| cheilinus_undulatus    | 499    | 29        | 0.995 | 0.968    |
| cromileptes_altivelis  | 499    | 30        | 0.995 | 0.945    |

</details>

## 🚗 Usage

### 🏁 Environment Setup

There are 2 options to install the development environment.

#### Option 1 - Developing Inside a Docker Container with Visual Studio Code's Dev Containers Extension (recommended):

- If you are using Windows, make sure that Windows Subsytem for Linux is installed and working on your machine (to do so, follow the instructions [here](https://learn.microsoft.com/en-us/windows/wsl/install)).
- Make sure Docker is installed on your machine.
- Install the Dev Containers Extension in Visual Studio Code (*ms-vscode-remote.remote-containers*).
- In VS Code, open the command palette (CTRL + SHIFT + P) and select *Dev Containers: Rebuild and Reopen in Container* (make sure Docker is running before executing this step). If the build seems to freeze, read the "Common Errors and Fixes" section below.

Note that the Dockerfile was created for CPU machines. If you wish to use GPU for inference, you can change the base image to `nvidia/cuda:12.0.0-runtime-ubuntu22.04`.

#### Option 2 - Developing on Your Host OS with Anaconda:

- Make sure Conda is installed and working on your machine (to do so, click [here](https://www.anaconda.com/download)).
- Then, run the following commands:
```shell
conda create --name your_env_name python=3.10
conda activate your_env_name
pip install -r requirements.txt
```

### 🚀 Inference

To make predictions on images or videos using the marine detection models, follow these simple steps:

1. **Models Download**: Refer to the 'Model Details' section for information on the models.
2. **Prediction Functions**: Utilize the following Python functions to generate predictions with bounding box annotations.

```python
from marine_detect.predict import predict_on_images, predict_on_video

# Predict on a set of images using FishInv and MegaFauna models
predict_on_images(
    model_paths=["path/to/FishInv/model", "path/to/MegaFauna/model"],
    confs_threshold=[0.522, 0.6],
    images_input_folder_path="path/to/input/images",
    images_output_folder_path="path/to/output/folder",
)

# Predict on a video using FishInv and MegaFauna models
predict_on_video(
    model_paths=["path/to/FishInv/model", "path/to/MegaFauna/model"],
    confs_threshold=[0.522, 0.6],
    input_video_path="path/to/input/video.mp4",
    output_video_path="path/to/output/video.mp4",
)
```
> [!NOTE]
> The optimal confidence thresholds for the FishInv model and the MegaFauna model are 0.522 and 0.6, respectively.

The resulting images or video files will have bounding boxes annotations, visually indicating the location and extent of the detected marine species within the original data. 

For example:

<p align="middle">
  <img src="./assets/images/input_folder/regq.jpg" width="300" />
  <img src="./assets/images/output_folder/regq.jpg" width="300" /> 
</p>

## 📚 References

### Datasets

- Ticon Dataset. (2023). Shark Dataset [ Open Source Dataset ]. In Roboflow Universe . Roboflow . https://universe.roboflow.com/ticon-dataset/shark-ibmby
- Minhajul Arefin. (2021).  zebra_shark Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/minhajul-arefin/zebra_shark
- Rizal Fadia Al Fikri. (2022).  shark_species Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/rizal-fadia-al-fikri/shark_species
- Aya Abd-Elnaser. (2022).  SHARK Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/aya-abd-elnaser/shark-jatfb
- Nomi. (2023).  seaturtle Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/nomi/seaturtle
- Parvej Hosen. (2022).  Turtle Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/parvej-hosen/turtle-f9xgw
- Seami New 5 Fishes. (2023).  EagleRay New Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/seami-new-5-fishes/eagleray-new
- Le Wagon. (2023).  count-a-manta Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/le-wagon-w02yl/count-a-manta
- Renaldo Rasfuldi. (2022).  fish_id_2 Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/renaldo-rasfuldi/fish_id_2
- Universiti Teknologi Malaysia. (2023).  Giant Clam Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/universiti-teknologi-malaysia-juyvx/giant-clam
- Universiti Teknologi Malaysia. (2023).  Tioman Giant Clams Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/universiti-teknologi-malaysia-juyvx/tioman-giant-clams
- Jacob Solawetz. (2023).  Fish Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/roboflow-gw7yv/fish-yzfml
- Dataset. (2022).  Dataset Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/dataset-gdypo/dataset-axhm3
- Addison Howard, W. K., Eunbyung Park. (2018). ImageNet Object Localization Challenge. Kaggle. https://kaggle.com/competitions/imagenet-object-localization-challenge
- Australian Institute of Marine Science (AIMS), University of Western Australia (UWA) and Curtin University. (2019), OzFish Dataset - Machine learning dataset for Baited Remote Underwater Video Stations, https://doi.org/10.25845/5e28f062c5097
- GBIF.org (09 January 2024) GBIF Occurrence Download  https://doi.org/10.15468/dl.w5xy62
- GBIF.org (09 January 2024) GBIF Occurrence Download  https://doi.org/10.15468/dl.a5uwzp
- GBIF.org (09 January 2024) GBIF Occurrence Download  https://doi.org/10.15468/dl.r5xqkc
- GBIF.org (09 January 2024) GBIF Occurrence Download  https://doi.org/10.15468/dl.ug7n62
- GBIF.org (19 December 2023) GBIF Occurrence Download  https://doi.org/10.15468/dl.32mwtb


### Model

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics