# Preprocessing Summary

## Selected Detection Classes

The project uses seven object detection classes from Cityscapes:

- person
- rider
- car
- truck
- bus
- motorcycle
- bicycle

## Why Preprocessing Was Required

Cityscapes stores annotations as polygon-based JSON files. YOLO models require one text file per image with normalized bounding box coordinates. Therefore, the raw annotations had to be converted before training.

## Preprocessing Steps Completed

1. Verified the dataset structure for clear Cityscapes, fine annotations, and Foggy Cityscapes.
2. Selected seven traffic-relevant instance classes for detection.
3. Parsed `gtFine` polygon annotation JSON files.
4. Filtered only the selected classes.
5. Converted polygons into bounding boxes using minimum and maximum x-y coordinates.
6. Converted bounding boxes into YOLO normalized format:
   - `class_id x_center y_center width height`
7. Built a YOLO-ready source dataset under `datasets/processed/source_yolo`.
8. Generated a dataset YAML file for training.
9. Validated the converted labels visually by drawing boxes on sample images.

## Output of Preprocessing

The main preprocessing outputs are stored under the `preprocessing` folder.

The main preprocessing outputs are:

- `preprocessing/source_yolo/images/train`
- `preprocessing/source_yolo/images/val`
- `preprocessing/source_yolo/images/test`
- `preprocessing/source_yolo/labels/train`
- `preprocessing/source_yolo/labels/val`
- `preprocessing/source_yolo/labels/test`
- `preprocessing/source_yolo/cityscapes_detection.yaml`
- `preprocessing/label_validation`

## Important Note About Test Split

The converted Cityscapes `test` split contains image files, but it should not be treated as a standard supervised evaluation split in the same way as train and val. For the project workflow, `train` and `val` are the main source-domain supervised splits.
