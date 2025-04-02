# Video OCR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a Python script that performs Optical Character Recognition (OCR) on video files on a frame
by frame basis and generates a csv file with the OCR strings and their confidence scores.

This was forked from [this repository](https://github.com/torbjornbp/video-ocr2srt) and simplified to read video
frames at a set interval and output the OCR strings to a csv file without requiring EAST. Converted to using EasyOCR.

Currently optimized to read numerals, such as to record readings from a meter over time.

### Current state

- Currently can output the following:
    - A CSV file that includes the OCR strings marked with frame number, timecode (in ms) and confidence scores for text
      detection and the actual OCR process.
    - An optional JSON file that includes some basic information about the file and parameters used, as well as OCR
      strings marked with frame number, timecode (in ms) and confidence scores for text detection and the actual OCR
      process.

### Prerequisites

- Python 3.8 or higher
- Libraries and packages: argparse, easyocr, cv2, imutils, pandas, tqdm

### Usage

```sh
python videoEasyOCR.py -v <path_to_video> -m <path_to_model> [-l <language>] [-f <frame_rate>] [-p]
```

Where:

- `<video>`: This argument is required and it should be the path to the video file you want to process.
- `<output>`: This argument is option and is the path to save the CSV. By default, it saves to the same directory as
  the video.
- `<language>`: This argument is optional and it should be the language model for Pytesseract. The default is `en` (
  English). You can provide other language codes supported by Tesseract.
- `<frame_rate>`: This argument is optional and it specifies the number of frames to skip for processing. By default,
  this value is `10`, which seems to give an ok compromise between detected text and and processing speed.
- `-p` or `--preview`: This argument is optional. If included, it enables a preview of the video.
- `-w` or `--whitelist`: This argument is optional. If included it lets you specify characters to whitelist in OCR. By
  default, numerals are whitelisted: `"0123456789"`
- `-j` or `--json`: This argument is optional. If included, it enables the output of a JSON file with the OCR 
  strings and  their confidence scores.
- "-c" or "--crop": This argument is optional. If included, it enables cropping of the video by selecting a region 
  with a mouse drag over the first frame. The region is saved and used for all subsequent frames.

To process a video file named `video.mp4`, you would use the following
command:

```sh
python videoEasyOCR.py -v video.mp4 -f 10
```

The script will process the video, performing OCR on every 10th frame and will output a csv in the format
`<video_filename>_<language>_<timestamp>.csv` with the OCR strings and their confidence scores.
