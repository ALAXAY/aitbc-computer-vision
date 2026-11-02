# AI Technologies (Basic Course): computer vision for UAV and UGV in Google Colab

Google Colab notebooks of the professional development course **AI Technologies (Basic Course)**, AIT-BC
(«Технології штучного інтелекту (базовий курс)»), 40 hours, developed at the Kruty Heroes Military Institute
of Telecommunications and Information Technology, Kyiv, Ukraine, 2026.

Author: **Oleksii Fesenko**, PhD, Associate Professor.

The notebooks are in English. Every notebook covers two or three lessons of the programme. Each block of code is
preceded by a text cell with the method, the formulas and the meaning of the parameters, and the parameters are
set through Colab forms (`#@param`). All examples use **real** imagery of UAVs and ground platforms (visible and
thermal frames, FPV video, annotated data sets), which is downloaded from the original sources at run time.

## Notebooks

| Notebook | Topic | Lessons | Open | Executed example |
|---|---|---|---|---|
| NB01 | 1 | 1.1 Lecture: main tasks of computer vision in UAV and UGV systems; 1.2 Group class: Google Colab tools for processing electro-optical imagery | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALAXAY/aitbc-computer-vision/blob/main/notebooks/AITBC_NB01_CV_tasks_and_Colab_tools.ipynb) | [view](executed_examples/AITBC_NB01_executed_example.ipynb) |
| NB02 | 1 | 1.3 Practical: setting up Google Colab and first work with frames of electro-optical systems; 1.4 Practical: processing of video streams of UAVs and ground robotic platforms | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALAXAY/aitbc-computer-vision/blob/main/notebooks/AITBC_NB02_Colab_frames_and_video.ipynb) | [view](executed_examples/AITBC_NB02_executed_example.ipynb) |
| NB03 | 2 | 2.1 Lecture: digital image and video stream of electro-optical surveillance devices; 2.2 Group class: stages of preprocessing of video frame data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALAXAY/aitbc-computer-vision/blob/main/notebooks/AITBC_NB03_Digital_image_and_preprocessing.ipynb) | [view](executed_examples/AITBC_NB03_executed_example.ipynb) |
| NB04 | 2 | 2.3 Practical: channel decomposition and geometric normalisation of a video frame; 2.4 Practical: filtering methods for the increase of the visual informativeness of video frames | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALAXAY/aitbc-computer-vision/blob/main/notebooks/AITBC_NB04_Channels_ROI_letterbox_and_filtering.ipynb) | [view](executed_examples/AITBC_NB04_executed_example.ipynb) |
| NB05 | 3 | 3.1 Lecture: extraction of edges, silhouettes and contours by classical methods; 3.3 Practical: silhouettes and thermal signatures in thermal frames; 3.4 Practical: stabilisation of the video stream of an FPV drone | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALAXAY/aitbc-computer-vision/blob/main/notebooks/AITBC_NB05_Edges_contours_and_FPV_stabilisation.ipynb) | [view](executed_examples/AITBC_NB05_executed_example.ipynb) |
| NB06 | 4 | 4.1 Lecture: machine learning in the recognition of military objects (data set, HOG, SVM and Random Forest, metrics); 4.2 Group class: from the neuron to MobileNet, transfer learning and ONNX | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALAXAY/aitbc-computer-vision/blob/main/notebooks/AITBC_NB06_Military_object_recognition_ML_and_CNN.ipynb) | [view](executed_examples/AITBC_NB06_executed_example.ipynb) |
| NB07 | 4 | 4.3 Practical: classification of thermal signatures by HOG + SVM, target versus background; 4.4 Practical: transfer learning of MobileNet for the recognition of military vehicles and export to ONNX | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALAXAY/aitbc-computer-vision/blob/main/notebooks/AITBC_NB07_Thermal_HOG_SVM_and_MobileNet_transfer_learning.ipynb) | [view](executed_examples/AITBC_NB07_executed_example.ipynb) |
| Practical | 1–5 | End-to-end practical notebook for the five topics: from a UAV video frame to detection and tracking | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALAXAY/aitbc-computer-vision/blob/main/notebooks/AITBC_Practical_Course_Colab_EN.ipynb) | [view](executed_examples/AITBC_Practical_Course_EN_executed_example.ipynb) |

**Status.** Notebooks NB01–NB07 cover 15 of the 19 lessons (30 of 40 hours). Topic 5 (lessons 5.1–5.4: detection
of objects by YOLO, segmentation of the traversable area, tracking, real-time video pipeline and the mini-project)
is in preparation as NB08 and NB09.

## How to work with the notebooks

1. Press **Open in Colab** next to a notebook, then *Runtime → Run all*, or run the cells one by one with `Shift+Enter`.
   Section 0 of each notebook is always run first.
2. The course library `aitbc.py` is embedded in every notebook, so no installation from this repository is needed.
   The file [`aitbc.py`](aitbc.py) in the root is the same library (version 2.6.0) for use outside Colab.
3. The data sets are downloaded at run time into `/content/aitbc_data`; a copy can be kept on Google Drive.
4. A GPU runtime (*Runtime → Change runtime type → T4 GPU*) is recommended for Lesson 4.4 of NB07; all other
   notebooks run on the CPU runtime.

The executed examples in [`executed_examples/`](executed_examples) were produced with the default parameters on a
local CPU computer. They show what the results look like before running a notebook.

## Repository structure

```
notebooks/            notebooks for Google Colab (without outputs)
executed_examples/    the same notebooks executed with the default parameters
figures/              figures of the executed examples, by notebook
aitbc.py              course library (embedded in the notebooks)
build/                scripts used to generate the notebooks
LICENSE, NOTICE       licence of the code and the text (Apache-2.0)
DATA_LICENSES.md      data sets, their authors and licences
CITATION.cff          how to cite the course
```

## Data sets

The repository does not contain data sets. The notebooks download them from the original publications and
repositories; the use of each data set is governed by its own licence, see [DATA_LICENSES.md](DATA_LICENSES.md).

| Data set | Content | Licence |
|---|---|---|
| HIT-UAV | thermal frames of a reconnaissance UAV, 60–130 m, day and night | CC BY 4.0 |
| Kust4K | pixel-aligned visible and thermal pairs over urban roads | CC BY 4.0 |
| M3OT | synchronised visible and thermal sequences of DJI Mavic 3T with vehicle tracks | CC BY 4.0 |
| GOOSE | camera images of a ground vehicle in unstructured terrain, 64 classes | CC BY-SA 4.0 |
| TII-RATM | FPV racing quadrotor frames with IMU and motion capture | CC BY 4.0 |
| SULAND / SULAND_v2 | frames with surrogates of surface landmines PFM-1 and PMA-2 | CC BY 4.0 |
| military_vehicles | photographs of 24 types of military vehicles | declared MIT, photographs from web sites |
| VisDrone2019-DET | UAV frames over cities | non-commercial use |

## Licence

The code and the text of the notebooks are licensed under the **Apache License 2.0**, see [LICENSE](LICENSE) and
[NOTICE](NOTICE). The licence does not extend to the data sets or to the figures that show their images: such
figures remain under the licence of the corresponding data set (see [DATA_LICENSES.md](DATA_LICENSES.md)).
Figures with frames of VisDrone and with photographs of the military_vehicles data set are omitted from the
executed examples in this repository; running the notebooks in Colab reproduces them.

## Citation

If you use these materials, please cite them as given in [CITATION.cff](CITATION.cff).
