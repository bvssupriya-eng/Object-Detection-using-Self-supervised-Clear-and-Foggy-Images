"""Class definitions used across the project."""

CITYSCAPES_DETECTION_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
]

CITYSCAPES_CLASS_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(CITYSCAPES_DETECTION_CLASSES)
}
