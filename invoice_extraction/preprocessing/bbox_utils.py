from typing import List 
def normalize_box(box: List[List[int]], width: int, height: int):
    """
    Function to normalize bounding boxes (4 points -> xmin, ymin, xmax, ymax).
    Convert 4-point box coordinates to [xmin, ymin, xmax, ymax] scaled to 0-1000.
    box format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return [
        int(1000 * (xmin / width)),
        int(1000 * (ymin / height)),
        int(1000 * (xmax / width)),
        int(1000 * (ymax / height)),
    ]
