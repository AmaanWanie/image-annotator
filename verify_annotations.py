import glob

ACTUATORS = {29, 30, 31, 32, 33, 34, 35, 74}
VALVES = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 41, 42, 43, 44, 67, 68, 70, 71, 198, 199, 200}

def yolo_to_bbox(x_center, y_center, width, height):
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    x_max = x_center + (width / 2)
    y_max = y_center + (height / 2)
    return x_min, y_min, x_max, y_max

def check_overlap(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    dx = min(x1_max, x2_max) - max(x1_min, x2_min)
    dy = min(y1_max, y2_max) - max(y1_min, y2_min)

    if (dx > 1e-6) and (dy > 1e-6): # Use a small epsilon for float comparison
        return True
    return False

def verify_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5: continue
        cls_id = int(parts[0])
        x_c, y_c, w, h = map(float, parts[1:])
        annotations.append({'cls_id': cls_id, 'bbox': yolo_to_bbox(x_c, y_c, w, h)})

    valves = [a for a in annotations if a['cls_id'] in VALVES]
    actuators = [a for a in annotations if a['cls_id'] in ACTUATORS]

    overlaps = 0
    for act in actuators:
        for valve in valves:
            if check_overlap(act['bbox'], valve['bbox']):
                overlaps += 1
    
    return overlaps

def main():
    train_labels = glob.glob('tiled_dataset/train/labels/*.txt')
    val_labels = glob.glob('tiled_dataset/val/labels/*.txt')
    all_files = train_labels + val_labels
    
    total_overlaps = 0
    files_with_overlaps = 0
    
    for f in all_files:
        overlaps = verify_file(f)
        if overlaps > 0:
            print(f"Overlap found in {f}: {overlaps}")
            total_overlaps += overlaps
            files_with_overlaps += 1
            
    if total_overlaps == 0:
        print("Verification Successful: No overlaps found.")
    else:
        print(f"Verification Failed: Found {total_overlaps} overlaps in {files_with_overlaps} files.")

if __name__ == "__main__":
    main()
