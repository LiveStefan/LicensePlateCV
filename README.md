# LicensePlateCV

Real-time license plate detection and OCR from the MacBook camera. The script detects a likely plate region, warps it, thresholds it, then runs Tesseract OCR and overlays the text + a confidence-like score on the video feed.

## Features
- Live webcam capture
- Plate-like contour detection + perspective warp
- OCR using Tesseract
- On-screen text overlay with a simple confidence estimate
- Debug windows for edges, warped plate, and threshold image

## Requirements
- Python 3.13
- macOS (camera capture)
- Tesseract OCR

## Install

### 1. Tesseract (macOS)
```bash
brew install tesseract
```

### 2. Python deps
```bash
pip install opencv-python imutils numpy pytesseract
```

If Tesseract isn’t found automatically, set the path in `video_stream.py`:
```python
# pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"
```

## Run
```bash
python video_stream.py
```

Press `q` to quit.

## Notes
- OCR quality depends heavily on lighting and camera angle.
- You can tweak Canny thresholds and contour filtering for better results.

## License
MIT
