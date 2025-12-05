import os
import glob
from PIL import Image, ImageDraw, ImageFont
import yaml
import numpy as np

# Define class IDs
ACTUATORS = {29, 30, 31, 32, 33, 34, 35, 74}
VALVES = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 41, 42, 43, 44, 67, 68, 70, 71, 198, 199, 200}

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def yolo_to_bbox(x_center, y_center, width, height, img_w, img_h):
    x_min = int((x_center - width / 2) * img_w)
    y_min = int((y_center - height / 2) * img_h)
    x_max = int((x_center + width / 2) * img_w)
    y_max = int((y_center + height / 2) * img_h)
    return x_min, y_min, x_max, y_max

def draw_boxes(image, annotations, class_names):
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    
    # Try to load a font, fallback to default if not found
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for ann in annotations:
        cls_id = ann['cls_id']
        if cls_id not in ACTUATORS and cls_id not in VALVES:
            continue
            
        x_c, y_c, w, h = ann['bbox']
        x_min, y_min, x_max, y_max = yolo_to_bbox(x_c, y_c, w, h, img_w, img_h)
        
        color = (0, 255, 0) if cls_id in VALVES else (255, 0, 0) # Green for valves, Red for actuators
        
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
        
        label = class_names.get(cls_id, str(cls_id))
        draw.text((x_min, y_min - 15), label, fill=color, font=font)
    return image

def process_dataset(split, class_names, suffix=''):
    img_dir = f'tiled_dataset/{split}/images'
    label_dir = f'tiled_dataset/{split}/labels'
    output_dir = f'data/annotations/{split}'
    
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = glob.glob(os.path.join(img_dir, '*'))
    print(f"Processing {len(img_files)} images in {split}...")
    
    for img_path in img_files:
        basename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(basename)[0]
        label_path = os.path.join(label_dir, name_without_ext + '.txt')
        
        if not os.path.exists(label_path):
            continue
            
        try:
            image = Image.open(img_path)
        except Exception as e:
            print(f"Could not open image {img_path}: {e}")
            continue
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        annotations = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5: continue
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:])
            annotations.append({'cls_id': cls_id, 'bbox': (x_c, y_c, w, h)})
            
        # Check if there are any relevant annotations before saving
        has_relevant = any(a['cls_id'] in ACTUATORS or a['cls_id'] in VALVES for a in annotations)
        
        if has_relevant:
            image = draw_boxes(image, annotations, class_names)
            
            # Construct output filename with suffix
            filename, ext = os.path.splitext(basename)
            out_name = f"{filename}{suffix}{ext}"
            image.save(os.path.join(output_dir, out_name))

def main():
    yaml_path = 'data.yaml'
    class_names = load_class_names(yaml_path)
    
    process_dataset('train', class_names)
    process_dataset('val', class_names)
    print("Visualization complete.")

if __name__ == "__main__":
    main()
