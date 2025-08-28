import uuid
from PIL import Image
def convert_yolo_to_box_data(yolo_file_path: str, image_path: str):
    """
    Converts YOLO format annotations to box_data format.

    Args:
        yolo_file_path (str): Path to the YOLO .txt file.
        image_path (str): Path to the corresponding image file.

    Returns:
        List[Dict]: List of box_data entries for this image.
    """
    box_data = []

    with open(yolo_file_path, "r") as f:
        lines = f.readlines()

    image = Image.open(image_path)
    img_width, img_height = image.size

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # invalid format

        class_id, x_center, y_center, w, h = map(float, parts)

        # Denormalize
        x = int((x_center - w / 2) * img_width)
        y = int((y_center - h / 2) * img_height)
        w_px = int(w * img_width)
        h_px = int(h * img_height)

        box_dict = {
            "id": str(uuid.uuid4()),
            "coords": (x, y, w_px, h_px),
            "lines": [],
            "user_lines": []
        }
        box_data.append(box_dict)

    return box_data

######################################
# Example usage
######################################
base_dir = r'F:\AI\MangaImages\LtoR\output'
new_boxes = convert_yolo_to_box_data(r"F:\AI\MangaImages\LtoR\output\label/1.txt", r"F:\AI\MangaImages\LtoR\output\img/1.jpg")
# self.boxes_data["images/img_01.jpg"] = new_boxes
# self.populate_text_list(new_boxes)
print(new_boxes)