# image-annotator
🚀 Ultra-Fast Bounding Box Viewer & Annotator

A high-performance image annotation tool built using OpenCV + Tkinter, designed for smooth panning, instant zooming, and accurate YOLO-format bounding box editing even on very large images (6000px – 12000px).

✨ Features
⚡ Ultra-Fast Rendering

Smooth panning at 60–120 FPS

Instant zoom (mouse wheel or buttons)

Handles very large JPGs (8000px+) with no lag

🖼️ Smart Image Pipeline

Loads full-resolution image for accurate YOLO saving

Uses optimized downscaled working image for real-time rendering

Uses OpenCV slicing for super-fast viewports

📝 Annotation Capabilities

Draw bounding boxes

Save annotations in YOLOv5/v8 format

Class ID displayed on top of each bbox

Add new boxes instantly

Annotation rectangle preview while dragging

Editable annotation directory

🎯 Bounding Box Overlay

BBoxes rendered live on Tkinter Canvas

Perfectly synchronized with pan & zoom

Works with any zoom level

No misalignment — even on 10k × 10k images
