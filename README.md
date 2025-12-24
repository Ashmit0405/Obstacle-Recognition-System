## CELL 1
```bash
!pip install -q ultralytics tqdm
!pip install -q mapillary-tools
!wget http://images.cocodataset.org/zips/train2017.zip
!unzip -q train2017.zip
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
!unzip -q annotations_trainval2017.zip
!pip install -q kaggle
!pip install -q roboflow
%env KAGGLE_KEY=
%env KAGGLE_USERNAME=
!kaggle datasets list
```
## CELL2
```bash
from roboflow import Roboflow

rf = Roboflow(api_key="")
project = rf.workspace("training-odujy").project("stairs-lusiz-hmjdf")
dataset = project.version(1).download("yolov8")
```

## CELL3
```bash
!python bdd100k_to_yolo.py
!python coco_to_safety_yolo.py
!python roboflow_merge.py
!python oversample_stairs.py
!python train-test-split.py
```

## CELL4
```bash
!yolo detect train \
  model=yolov8n.pt \
  data=safety.yaml \
  epochs=30 \
  imgsz=640 \
  batch=16 \
  lr0=1e-4 \
  cls=2.0
```