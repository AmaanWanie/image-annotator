import tkinter as tk
from tkinter import ttk, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import glob
import time


MAX_WORKING_SIZE = 2000  # Working resolution for fast rendering


class FastBBoxViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultra-Fast BBox Viewer (OpenCV Engine)")
        self.root.geometry("1400x900")

        # Paths
        self.img_dir = r"val/images"
        self.label_dir = r"val/labels"

        # Image list
        self.image_files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        self.current_idx = 0

        # Zoom & Pan
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start = None

        # Throttle
        self.last_render = 0

        # Annotation
        self.annotation_mode = False
        self.rect_start = None
        self.rect_end = None

        # Image storage
        self.original = None        # Full resolution (numpy)
        self.working = None         # Scaled (numpy)
        self.zoomed = None          # Zoomed (numpy)
        self.tk_image = None
        self.bboxes = []            # (cls, xc, yc, w, h)

        # Canvas
        self.canvas = tk.Canvas(root, bg="gray", cursor="hand2")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_scroll)

        # Controls
        frame = ttk.Frame(root)
        frame.pack(fill=tk.X)

        ttk.Button(frame, text="Prev", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(frame, text="Next", command=self.next_image).pack(side=tk.LEFT)
        ttk.Button(frame, text="Zoom+", command=self.zoom_in).pack(side=tk.LEFT)
        ttk.Button(frame, text="Zoom-", command=self.zoom_out).pack(side=tk.LEFT)
        ttk.Button(frame, text="Reset", command=self.reset_view).pack(side=tk.LEFT)
        ttk.Button(frame, text="Annotate", command=self.toggle_annotate).pack(side=tk.LEFT)

        self.info = ttk.Label(frame, text="")
        self.info.pack(side=tk.LEFT, padx=20)

        self.zoom_label = ttk.Label(frame, text="Zoom: 100%")
        self.zoom_label.pack(side=tk.LEFT)

        self.load_image()


    # --------------------------------------------------------------------------
    # IMAGE LOADING
    # --------------------------------------------------------------------------

    def load_image(self):
        img_path = self.image_files[self.current_idx]

        # Load full resolution OpenCV image
        self.original = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Create working image for speed
        h, w = self.original.shape[:2]
        scale = MAX_WORKING_SIZE / max(h, w)
        if scale < 1:
            self.working = cv2.resize(self.original, (int(w * scale), int(h * scale)), cv2.INTER_LINEAR)
        else:
            self.working = self.original.copy()

        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        # Load bboxes
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, name + ".txt")
        self.bboxes = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    cls, xc, yc, w, h = line.strip().split()
                    self.bboxes.append((int(cls), float(xc), float(yc), float(w), float(h)))

        self.info.config(text=f"{self.current_idx+1}/{len(self.image_files)} - {os.path.basename(img_path)}")
        self.apply_zoom()


    # --------------------------------------------------------------------------
    # ZOOM & PAN SYSTEM
    # --------------------------------------------------------------------------

    def apply_zoom(self):
        """Recompute zoomed working image."""
        h, w = self.working.shape[:2]
        zw = int(w * self.zoom)
        zh = int(h * self.zoom)
        self.zoomed = cv2.resize(self.working, (zw, zh), cv2.INTER_LINEAR)

        self.render_view()

    def render_view(self):
        """Crop zoomed image based on pan and draw canvas + bboxes."""

        # throttle to ~60 fps
        if time.time() - self.last_render < 0.015:
            return
        self.last_render = time.time()

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        if cw < 10 or ch < 10:
            return

        # clamp pan
        max_x = max(0, self.zoomed.shape[1] - cw)
        max_y = max(0, self.zoomed.shape[0] - ch)
        self.pan_x = max(0, min(self.pan_x, max_x))
        self.pan_y = max(0, min(self.pan_y, max_y))

        # crop viewport
        view = self.zoomed[self.pan_y:self.pan_y+ch, self.pan_x:self.pan_x+cw]

        # Convert to Tk image
        pil_img = Image.fromarray(view)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # draw bboxes
        self.draw_bboxes()

        # draw temp rectangle if annotating
        if self.rect_start and self.rect_end:
            sx = self.rect_start[0] - self.pan_x
            sy = self.rect_start[1] - self.pan_y
            ex = self.rect_end[0] - self.pan_x
            ey = self.rect_end[1] - self.pan_y
            self.canvas.create_rectangle(sx, sy, ex, ey, outline="green", width=2)

        self.zoom_label.config(text=f"Zoom: {int(self.zoom*100)}%")

    # --------------------------------------------------------------------------
    # DRAW BBOXES
    # --------------------------------------------------------------------------
    
        # --------------------------------------------------------------------------
    # DRAW BBOXES
    # --------------------------------------------------------------------------
    def draw_bboxes(self):
        """Draw YOLO bboxes scaled to the zoomed working image."""
        ow, oh = self.original.shape[1], self.original.shape[0]
        ww, wh = self.working.shape[1], self.working.shape[0]

        scale_w = self.zoomed.shape[1] / ww
        scale_h = self.zoomed.shape[0] / wh

        for cls, xc, yc, w, h in self.bboxes:
            # convert YOLO → original px
            x1 = (xc - w/2) * ow
            y1 = (yc - h/2) * oh
            x2 = (xc + w/2) * ow
            y2 = (yc + h/2) * oh

            # convert original → working
            dx1 = x1 * (ww / ow)
            dy1 = y1 * (wh / oh)
            dx2 = x2 * (ww / ow)
            dy2 = y2 * (wh / oh)

            # apply zoom
            zx1 = dx1 * scale_w - self.pan_x
            zy1 = dy1 * scale_h - self.pan_y
            zx2 = dx2 * scale_w - self.pan_x
            zy2 = dy2 * scale_h - self.pan_y

            # draw rectangle
            self.canvas.create_rectangle(zx1, zy1, zx2, zy2, outline="red", width=2)

            # draw class id text above the box
            text_x = zx1 + 3
            text_y = zy1 - 15  # slightly above the bbox

            self.canvas.create_text(
                text_x, text_y,
                text=str(cls),
                fill="red",
                anchor="nw",
                font=("Arial", 12, "bold")
            )


    # def draw_bboxes(self):
    #     """Draw YOLO bboxes scaled to the zoomed working image."""
    #     ow, oh = self.original.shape[1], self.original.shape[0]
    #     ww, wh = self.working.shape[1], self.working.shape[0]

    #     scale_w = self.zoomed.shape[1] / ww
    #     scale_h = self.zoomed.shape[0] / wh

    #     for cls, xc, yc, w, h in self.bboxes:
    #         # convert YOLO → original px
    #         x1 = (xc - w/2) * ow
    #         y1 = (yc - h/2) * oh
    #         x2 = (xc + w/2) * ow
    #         y2 = (yc + h/2) * oh

    #         # convert original → working
    #         dx1 = x1 * (ww / ow)
    #         dy1 = y1 * (wh / oh)
    #         dx2 = x2 * (ww / ow)
    #         dy2 = y2 * (wh / oh)

    #         # apply zoom
    #         zx1 = dx1 * scale_w - self.pan_x
    #         zy1 = dy1 * scale_h - self.pan_y
    #         zx2 = dx2 * scale_w - self.pan_x
    #         zy2 = dy2 * scale_h - self.pan_y

    #         self.canvas.create_rectangle(zx1, zy1, zx2, zy2, outline="red", width=2)

    # --------------------------------------------------------------------------
    # MOUSE EVENTS
    # --------------------------------------------------------------------------

    def on_click(self, e):
        if self.annotation_mode:
            self.rect_start = (self.pan_x + e.x, self.pan_y + e.y)
            self.rect_end = self.rect_start
        else:
            self.drag_start = (e.x, e.y)

    def on_drag(self, e):
        if self.annotation_mode:
            self.rect_end = (self.pan_x + e.x, self.pan_y + e.y)
            self.render_view()
        else:
            dx = e.x - self.drag_start[0]
            dy = e.y - self.drag_start[1]
            self.pan_x -= dx
            self.pan_y -= dy
            self.drag_start = (e.x, e.y)
            self.render_view()

    def on_release(self, e):
        if self.annotation_mode and self.rect_start and self.rect_end:
            self.save_annotation()
        self.drag_start = None

    def on_scroll(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    # --------------------------------------------------------------------------
    # SAVE ANNOTATION
    # --------------------------------------------------------------------------

    def save_annotation(self):
        class_id = simpledialog.askinteger("Class ID", "Enter class ID (0–239):")
        if class_id is None:
            self.rect_start = None
            self.rect_end = None
            return

        # Working → original mapping
        ww, wh = self.working.shape[1], self.working.shape[0]
        ow, oh = self.original.shape[1], self.original.shape[0]

        sx, sy = self.rect_start
        ex, ey = self.rect_end

        # Convert zoomed coords → working coords
        scale_w = self.zoomed.shape[1] / ww
        scale_h = self.zoomed.shape[0] / wh

        dx1 = sx / scale_w
        dy1 = sy / scale_h
        dx2 = ex / scale_w
        dy2 = ey / scale_h

        # working → original
        x1 = dx1 * (ow / ww)
        y1 = dy1 * (oh / wh)
        x2 = dx2 * (ow / ww)
        y2 = dy2 * (oh / wh)

        # YOLO
        xc = (x1 + x2) / 2 / ow
        yc = (y1 + y2) / 2 / oh
        w = abs(x2 - x1) / ow
        h = abs(y2 - y1) / oh

        name = os.path.splitext(os.path.basename(self.image_files[self.current_idx]))[0]
        label_path = os.path.join(self.label_dir, name + ".txt")
        os.makedirs(self.label_dir, exist_ok=True)

        with open(label_path, "a") as f:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        self.bboxes.append((class_id, xc, yc, w, h))
        self.rect_start = None
        self.rect_end = None
        self.render_view()

    # --------------------------------------------------------------------------
    # NAV
    # --------------------------------------------------------------------------

    def next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.load_image()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def toggle_annotate(self):
        self.annotation_mode = not self.annotation_mode
        self.canvas.config(cursor="crosshair" if self.annotation_mode else "hand2")
        self.rect_start = None
        self.rect_end = None

    def zoom_in(self):
        self.zoom *= 1.2
        self.apply_zoom()

    def zoom_out(self):
        self.zoom /= 1.2
        self.zoom = max(0.1, self.zoom)
        self.apply_zoom()

    def reset_view(self):
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.apply_zoom()


if __name__ == "__main__":
    root = tk.Tk()
    app = FastBBoxViewer(root)
    root.mainloop()
