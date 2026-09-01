from video_retrieval.models import ObjectRequirement


COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

COCO_LABEL_SET = frozenset(COCO_LABELS)


def parse_object_requirements(value: object) -> list[ObjectRequirement]:
    if not isinstance(value, list):
        return []
    requirements: list[ObjectRequirement] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            label = item.strip().lower()
            min_count = 1
        elif isinstance(item, dict):
            label = str(item.get("label") or "").strip().lower()
            try:
                min_count = max(1, int(item.get("min_count") or 1))
            except (TypeError, ValueError):
                min_count = 1
        else:
            continue
        if label not in COCO_LABEL_SET or label in seen:
            continue
        requirements.append(ObjectRequirement(label=label, min_count=min_count))
        seen.add(label)
    return requirements
