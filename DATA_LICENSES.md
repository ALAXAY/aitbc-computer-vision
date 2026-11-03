# Data sets used in the notebooks and their licences

The repository does not redistribute any data set. The notebooks download the data at run time from the sources
below; whoever runs the notebooks is responsible for complying with the licence of each data set.
The Apache License 2.0 of this repository applies to the code and the text only.

| Data set | Notebooks | Reference | Licence | Figures in this repository |
|---|---|---|---|---|
| HIT-UAV | NB01–NB05, NB07, NB09, practical | Suo J. et al. HIT-UAV: a high-altitude infrared thermal dataset for unmanned aerial vehicle-based object detection. *Scientific Data* 10, 227 (2023) | CC BY 4.0 | included, CC BY 4.0 |
| Kust4K | NB01–NB05 | *Scientific Data* 12, 1701 (2025) | CC BY 4.0 | included, CC BY 4.0 |
| M3OT | NB02–NB05, NB08, NB09, practical | Nie Z. et al. *Scientific Data* 12, 1927 (2025) | CC BY 4.0 | included, CC BY 4.0 |
| GOOSE | NB01–NB04, NB08 | Mortimer P. et al. The GOOSE dataset for perception in unstructured environments. IEEE ICRA 2024 | CC BY-SA 4.0 | included, CC BY-SA 4.0 |
| TII-RATM | NB05, NB09 | Bosello M. et al. *IEEE Robotics and Automation Letters* 9(4), 3799–3806 (2024) | CC BY 4.0 | included, CC BY 4.0 |
| SULAND | NB06 | Vivoli E., Bertini M., Capineri L. Deep learning-based real-time detection of surface landmines using optical imaging. *Remote Sensing* 16(4), 677 (2024), doi:10.3390/rs16040677 | CC BY 4.0 | included, CC BY 4.0 |
| SULAND_v2 (refined annotation) | NB06 | Lekhak S. et al. arXiv:2607.28996 (2026); Hugging Face `SagarLekhak/SULAND_v2_RGB_Surface_Landmine_Dataset` | CC BY 4.0 | included, CC BY 4.0 |
| military_vehicles | NB06, NB07 | Kricheli J. S. et al. arXiv:2407.15192 (2024); Hugging Face `leibnitz-lab/military_vehicles` | MIT declared by the authors of the data set; the photographs were collected from web sites | photographs omitted; charts and confusion matrices included |
| VisDrone2019-DET | NB01–NB04, practical | Zhu P. et al. Detection and tracking meet drones challenge. *IEEE TPAMI* 44(11), 7380–7399 (2022) | non-commercial use (CC BY-NC-SA 3.0 IGO on public mirrors) | frames omitted; charts included |
| DUT Anti-UAV | NB08, NB09 | Zhao J., Zhang J., Li D., Wang D. Vision-based Anti-UAV Detection and Tracking. *IEEE Transactions on Intelligent Transportation Systems* (2022), arXiv:2205.10851; https://github.com/wangdongdut/DUT-Anti-UAV, Hugging Face copy `aegean-ai/dut-anti-uav` | Apache-2.0 | included, Apache-2.0 |

## Omitted figures

In the executed examples, an output that showed frames of VisDrone or photographs of the military_vehicles data set
is replaced by the note *"Figure omitted in the public copy of the repository"*. Running the cell in Google Colab
reproduces the figure from the original data. Statistics, charts and confusion matrices computed from these data
sets are kept.

## Pretrained models and software installed at run time

| Component | Use | Licence |
|---|---|---|
| torchvision models with ImageNet weights (MobileNetV3 and others), LR-ASPP MobileNetV3 with COCO weights | NB06, NB07, NB08, practical | BSD-3-Clause (code); terms of use of ImageNet and COCO for the training data |
| OpenCV, ONNX, ONNX Runtime | all notebooks | Apache-2.0 / MIT |
| PyTorch, scikit-learn, scikit-image | all notebooks | BSD-3-Clause |
| Ultralytics YOLO11 and its weights | NB08, NB09, practical (optional cells) | AGPL-3.0; installed from PyPI at run time and not redistributed here. The detector trained in Lesson 5.4 is a derived work of YOLO11 under the same licence |
