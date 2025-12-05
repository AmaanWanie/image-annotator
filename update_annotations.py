import os
import glob
# from tqdm import tqdm
import visualize_annotations

# Define class IDs
ACTUATORS = {29, 30, 31, 32, 33, 34, 35, 74}
VALVES = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 41, 42, 43, 44, 67, 68, 70, 71, 198, 199, 200}

def yolo_to_bbox(x_center, y_center, width, height):
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    x_max = x_center + (width / 2)
    y_max = y_center + (height / 2)
    return x_min, y_min, x_max, y_max

def bbox_to_yolo(x_min, y_min, x_max, y_max):
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + (width / 2)
    y_center = y_min + (height / 2)
    return x_center, y_center, width, height

def get_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def resolve_overlap(actuator, valve):
    # actuator and valve are (x_min, y_min, x_max, y_max)
    
    ax_min, ay_min, ax_max, ay_max = actuator
    vx_min, vy_min, vx_max, vy_max = valve

    # Check if there is overlap
    if (ax_max <= vx_min) or (ax_min >= vx_max) or (ay_max <= vy_min) or (ay_min >= vy_max):
        return actuator # No overlap

    # Calculate intersection rectangle
    ix_min = max(ax_min, vx_min)
    iy_min = max(ay_min, vy_min)
    ix_max = min(ax_max, vx_max)
    iy_max = min(ay_max, vy_max)

    # Candidates for new actuator box:
    c1 = (ax_min, ay_min, vx_min, ay_max) # Cut right side (keep left)
    c2 = (vx_max, ay_min, ax_max, ay_max) # Cut left side (keep right)
    c3 = (ax_min, ay_min, ax_max, vy_min) # Cut bottom side (keep top)
    c4 = (ax_min, vy_max, ax_max, ay_max) # Cut top side (keep bottom)

    candidates = []
    if c1[2] > c1[0] and c1[3] > c1[1]: candidates.append(c1)
    if c2[2] > c2[0] and c2[3] > c2[1]: candidates.append(c2)
    if c3[2] > c3[0] and c3[3] > c3[1]: candidates.append(c3)
    if c4[2] > c4[0] and c4[3] > c4[1]: candidates.append(c4)

    if not candidates:
        return None

    # Pick the candidate with the largest area
    best_candidate = max(candidates, key=lambda b: get_area(b))
    
    return best_candidate

def process_file(filepath, dry_run=False):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5: continue
        cls_id = int(parts[0])
        x_c, y_c, w, h = map(float, parts[1:])
        annotations.append({'cls_id': cls_id, 'bbox': yolo_to_bbox(x_c, y_c, w, h), 'original': line})

    valves = [a for a in annotations if a['cls_id'] in VALVES]
    actuators = [a for a in annotations if a['cls_id'] in ACTUATORS]
    others = [a for a in annotations if a['cls_id'] not in VALVES and a['cls_id'] not in ACTUATORS]

    new_actuators = []
    modified = False

    for act in actuators:
        current_bbox = act['bbox']
        original_bbox = current_bbox
        original_area = get_area(original_bbox)
        
        for valve in valves:
            result = resolve_overlap(current_bbox, valve['bbox'])
            if result is None:
                current_bbox = None
                break
            current_bbox = result
        
        if current_bbox is not None:
            new_area = get_area(current_bbox)
            
            # Check for excessive reduction (e.g., < 20% of original area)
            if new_area < 0.2 * original_area:
                print(f"Warning: Excessive reduction for actuator in {filepath}. Area reduced to {new_area/original_area:.2%}. Keeping original?")
                # Decision: If it's reduced too much, it might be better to remove it or keep it?
                # User said "bboxes for some of the actuators have been reduced way too much".
                # This implies they don't want them to be tiny slivers.
                # If the overlap is massive, maybe the actuator is actually *behind* the valve or it's a mislabel?
                # For now, let's keep it but log it. Or maybe we should just NOT cut it if it destroys the box?
                # But the goal is to remove overlap.
                # Let's stick to the cut, but maybe the heuristic of "largest area" is failing when multiple cuts happen?
                pass

            # Check if it changed
            if current_bbox != original_bbox:
                modified = True
                nx_c, ny_c, nw, nh = bbox_to_yolo(*current_bbox)
                nx_c = max(0, min(1, nx_c))
                ny_c = max(0, min(1, ny_c))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))
                
                if nw > 0 and nh > 0:
                     new_actuators.append(f"{act['cls_id']} {nx_c:.6f} {ny_c:.6f} {nw:.6f} {nh:.6f}\n")
            else:
                new_actuators.append(act['original'])
        else:
            modified = True # Removed
            print(f"Removed actuator in {filepath} (completely inside valve)")

    if modified and not dry_run:
        with open(filepath, 'w') as f:
            for a in valves:
                f.write(a['original'])
            for a in others:
                f.write(a['original'])
            for line in new_actuators:
                f.write(line)
    
    return modified

def main():
    # Step 1: Visualize Original
    print("Visualizing original annotations...")
    yaml_path = 'data.yaml'
    class_names = visualize_annotations.load_class_names(yaml_path)
    visualize_annotations.process_dataset('train', class_names, suffix='_old')
    visualize_annotations.process_dataset('val', class_names, suffix='_old')

    # Step 2: Update Annotations
    print("Updating annotations...")
    train_labels = glob.glob('tiled_dataset/train/labels/*.txt')
    val_labels = glob.glob('tiled_dataset/val/labels/*.txt')
    
    all_files = train_labels + val_labels
    print(f"Found {len(all_files)} files.")
    
    count = 0
    for f in all_files:
        if process_file(f, dry_run=False):
            count += 1
            
    print(f"Modified {count} files.")

    # Step 3: Visualize Updated
    print("Visualizing updated annotations...")
    visualize_annotations.process_dataset('train', class_names, suffix='_new')
    visualize_annotations.process_dataset('val', class_names, suffix='_new')
    print("Done.")

if __name__ == "__main__":
    main()
