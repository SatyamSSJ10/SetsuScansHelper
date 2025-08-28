# There is Error in Normalize coords or Is Inside

SIZE = 1024

def normalize_coords(coords, img_w=1024, img_h=1024):
    """Convert absolute coordinates to YOLO format (normalized x_center,y_center, width, height)"""
    x1, y1, w, h = coords
    x_center = (x1 + w / 2) / img_w
    y_center = (y1 + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    
    return [x_center, y_center, width, height]

def panel_sequencer(sequencer, bubble_coords):
    sequence = sequencer.predict(bubble_coords)
    # next = [abs(x - max(sequence)) for x in sequence]
    # print(next)
    return sequence

def ai_sort_panel(model,bubbles): # L to R
    if len(bubbles) <= 1:
        return bubbles
    
    if len(bubbles) == 2:
        return sorted(bubbles, key=lambda x: x["coords"][1]) 
    
    coord_list = [y["coords"] for y in bubbles]
    sequence = panel_sequencer(model, coord_list)
    order_mapping = {idx: seq_idx for seq_idx, idx in enumerate(sequence)}
    return sorted(bubbles, key= lambda x: order_mapping[bubbles.index(x)]) 

def ai_sort_bubble(model,bubbles): # L to R
    if len(bubbles) <= 1:
        return bubbles
    
    # if len(bubbles) == 2:
    #     return sorted(bubbles, key=lambda x: (x["coords"][0] + x["coords"][2]//2, x["coords"][1]+ x["coords"][3]//2)) 
    
    coord_list = [normalize_coords(y["coords"]) for y in bubbles]
    sequence = panel_sequencer(model, coord_list)
    order_mapping = {idx: seq_idx for seq_idx, idx in enumerate(sequence)}
    return sorted(bubbles, key= lambda x: order_mapping[bubbles.index(x)]) 

def organize_bubbles(file_data, yolo_panels, model, image_size):
    
    def is_inside( id, bubble_coords, panel_coords, image_size):
        bx, by, bw, bh = bubble_coords # Is not normalized
        pxc, pyc, pw, ph = panel_coords # Is normalized
        px1 = int((pxc - pw / 2) * image_size[0])
        py1 = int((pyc - ph / 2) * image_size[1])
        pw = int(pw * image_size[0])
        ph = int(ph * image_size[1])
        return (bx+bw//2 >= px1 and by+bh//2 >= py1 and bx + bw//2 <= px1 + pw and by + bh//2 <= py1 + ph)
        

    # Step 1: Categorize bubbles into panels
    panels = []
    
    # Initialize panels from YOLO detection
    for idx, yolo_panel in enumerate(yolo_panels):
        panel = {
            "id": idx,
            "coords": yolo_panel,
            "lines": []
        }
        panels.append(panel)

    # Assign bubbles to panels or create new panels
    for bubble in file_data:
        assigned = False
        # Check against all existing panels (YOLO and generated)
        for panel in panels:
            if is_inside(panel["id"],bubble["coords"], panel["coords"], image_size):
                panel["lines"].append(bubble)
                assigned = True
                break
        if not assigned:
            # Create new panel using bubble's coordinates
            normalized_coords = normalize_coords(bubble["coords"])
            is_inside(panel["id"],bubble["coords"], panel["coords"], image_size)
            new_panel = {
                "id": len(panels),
                "coords": normalized_coords,
                "lines": [bubble]
            }
            panels.append(new_panel)
    
    # Step 3: Sort panels using custom AI
    sorted_panels = ai_sort_panel(model, panels)

    # Step 2: Sort bubbles within each panel
    for panel in panels:
        if panel["lines"]:
            panel["lines"] = ai_sort_bubble(model, panel["lines"])
    
    # Step 4: Flatten into original format
    sorted_bubbles = []
    for panel in sorted_panels:
        sorted_bubbles.extend(panel["lines"])
    
    return sorted_bubbles