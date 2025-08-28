import os
import json
from PIL import Image
from typing import List, Dict


def export_bounding_boxes_to_yolo(output_dir: str, class_id, data):
    """
    Exports all bounding boxes from the current image to YOLO format files.

    Args:
        output_dir (str): Directory to save the .txt files.
        class_id (int): Class ID to assign to each bounding box (default is 0).
    """
    image_files = data.keys()
    boxes_data = data
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for file_path in image_files:
        file_data = boxes_data.get(file_path, [])
        if not file_data:
            continue

        image = Image.open(file_path)
        img_width, img_height = image.size

        # Generate YOLO lines
        yolo_lines = []
        for class_id,box in enumerate(file_data):
            x, y, w, h = box["coords"]
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_lines.append(yolo_line)

        # Write to file with same basename
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        yolo_file = os.path.join(output_dir, base_filename + ".txt")

        with open(yolo_file, "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"[INFO] Exported YOLO files to {output_dir}")

if __name__ == "__main__":
    # Example usage
    # Assuming `self.image_files` and `self.boxes_data` are already populated
    directory = "F:\AI\MangaImages\LtoR"
    json_path = os.path.join(directory, "annotations.json")
    if os.path.exists(json_path):
            with open(json_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                # data: { file_path: [ { "id":..., "coords":..., "lines":[...] }, ... ], ... }
    export_bounding_boxes_to_yolo(output_dir="./yolo_labels", class_id=0, data=data)