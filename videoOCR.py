import argparse
import json
import pytesseract
import cv2
import pandas as pd
from datetime import datetime
from tqdm import tqdm

cropped = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropped

    # Record the starting (x, y) coordinates on left mouse button down
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start= x, y

    # Record the ending (x, y) coordinates on left mouse button up
    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropped = True

def main(args):
    global cropped, x_start, y_start, x_end, y_end

    # Define paths for the video file and the pre-trained model file.
    videoFilePath = args.video
    outputFilePath = args.output if args.output else videoFilePath

    # Configuring Pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseractLanguage = args.language

    # Configuring Pytesseract to whitelist letters
    pytesseractWhitelist = args.whitelist

    # Open the video file.
    stream = cv2.VideoCapture(videoFilePath)

    # Fetch video properties.
    video_fps = stream.get(cv2.CAP_PROP_FPS)
    total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    tDelta = round(args.frame_rate / video_fps, 2)
    processingFrames = int(total_frames / args.frame_rate)

    print(f"Video FPS: {video_fps}, Video Duration: {round(total_frames / video_fps, 2)} seconds")
    print(f"Processing OCR every {args.frame_rate} frames (every {tDelta} seconds), "
          f"with {processingFrames} frames to be processed in total")

    # Initiate some variables.
    frame_count = -args.frame_rate
    progress_bar = tqdm(total=processingFrames, unit='frames')

    # Create an empty list to hold the data
    entries = {'frame_number': [], 'timecode_ms': [], 'ocr_text': [], 'ocr_confidence': []}

    # Create an empty list to hold the JSON data.
    json_output = []

    # Main loop for processing the video frames.
    while True:
        # Applying OCR to every nth frame of the video, where n is defined by args.frame_rate.
        # TODO: implement custom video start time
        frame_count += args.frame_rate
        stream.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        ret, frame = stream.read()
        if not ret:
            break

        orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the first frame and set up the mouse callback
        if args.crop and cropped is False:
            cv2.namedWindow("Select ROI")
            cv2.setMouseCallback("Select ROI", mouse_crop, 0)

            while cropped is False:
                cv2.imshow("Select ROI", orig)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if cropped:
                    cv2.destroyWindow("Select ROI")

        if args.crop:
            orig = orig[y_start:y_end, x_start:x_end]

        text = pytesseract.image_to_string(orig, config=f"-l {pytesseractLanguage} --oem 1 --psm 3 -c "
                                                        f"tessedit_char_whitelist={pytesseractWhitelist}")
        # Extract OCR confidence score
        data = pytesseract.image_to_data(orig, config=f"-l {pytesseractLanguage} --oem 1 --psm 3",
                                         output_type=pytesseract.Output.DICT)
        confidence_scores = [int(conf) for conf in data['conf'] if str(conf).isdigit()]
        confidence_tesseract = sum(confidence_scores) / len(confidence_scores) if confidence_scores else -1.0

        # Define the timing and length for each subtitle object.
        start_time_ms = stream.get(cv2.CAP_PROP_POS_MSEC)
        end_time_ms = start_time_ms + (tDelta * 1000)

        # Append data to dict
        entries['frame_number'].append(frame_count)
        entries['timecode_ms'].append(start_time_ms)
        entries['ocr_text'].append(str(text).strip())
        entries['ocr_confidence'].append(confidence_tesseract)

        # Append data to json_output list
        json_output.append({
            'frame_number': frame_count,
            'timecode_ms': start_time_ms,
            'ocr_text': text,
            'ocr_confidence': confidence_tesseract  # The confidence of the OCR string
        })

        # Display a video preview with bounding boxes if the preview is enabled.
        if args.preview:
            cv2.imshow("Preview", orig)

        # Update progress bar.
        progress_bar.update()

        # Exit the loop if 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Close progress bar, release the video file and destroy all windows.
    progress_bar.close()
    stream.release()
    cv2.destroyAllWindows()

    if args.json:
        # Create a dictionary to hold additional JSON information
        extra_info = {
            'filename': videoFilePath,
            'date_processed': datetime.now().strftime("%Y-%m-%d-%H-%M"),
            'ocr_language': pytesseractLanguage,
            'analysis_frame_interval': args.frame_rate,
            'character_whitelist': args.whitelist
        }

        # Insert the extra_info dictionary at the beginning of the json_output list
        json_output.insert(0, extra_info)

        # Define the output filename for the JSON file.
        output_json_filename = outputFilePath.rsplit('.', 1)[
                                   0] + "_" + pytesseractLanguage + "_" + datetime.now().strftime(
            "%Y-%m-%d-%H-%M") + ".json"
        print(f"Preparing to write JSON to file: {output_json_filename}")

        # Try to write JSON data to the file.
        try:
            with open(output_json_filename, 'w') as json_file:
                json.dump(json_output, json_file, indent=4)
            print("JSON file written successfully")
        except Exception as e:
            print(f"Error while writing JSON file: {e}")

    # Write dataframe to csv
    output_csv_filename = outputFilePath.rsplit('.', 1)[0] + "_" + pytesseractLanguage + "_" + datetime.now().strftime(
        "%Y-%m-%d-%H-%M") + ".csv"
    print(f"Preparing to write CSV to file: {output_csv_filename}")
    pd.DataFrame.from_dict(entries).to_csv(output_csv_filename, index=True)
    print("CSV file written successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract text from video using OCR and generate SRT file')
    parser.add_argument('-v', '--video', help='Path to the video file', required=True)
    parser.add_argument('-o', '--output', help='Path to the output file')
    parser.add_argument('-l', '--language', help='Language model for Pytesseract', default='eng')
    parser.add_argument('-f', '--frame_rate', help='Number of frames to skip for processing', type=int, default=10)
    parser.add_argument('-p', '--preview', help='Enable preview of the video', action='store_true')
    parser.add_argument('-w', '--whitelist', help='whitelist characters to improve OCR result',
                        default='1234567890')
    parser.add_argument('-j', '--json', help='Enable JSON output', action='store_true')
    parser.add_argument('-c', '--crop',
                        help='Crop the video frame by dragging selected region with cursor', action='store_true')

    args = parser.parse_args()

    main(args)
