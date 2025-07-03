import argparse
import easyocr
import json
import cv2
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm


def filter_text(text, whitelist):
    return ''.join([char for char in text if char in whitelist])


def main(args):
    # Define paths for the video file
    videoFilePath = args.video
    videoName = os.path.splitext(os.path.basename(videoFilePath))[0]
    outputFolderPath = args.output if args.output else os.path.dirname(videoFilePath)

    # Setup EasyOCR reader
    reader = easyocr.Reader([args.language])

    # Open the video file.
    if not os.path.isfile(videoFilePath):
        exit("Video file not found")
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

    # Create roi
    roi = False

    # Main loop for processing the video frames.
    while True:
        # Applying OCR to every nth frame of the video, where n is defined by args.frame_rate.
        # TODO: implement custom video start time
        frame_count += args.frame_rate

        if frame_count >= total_frames:
            break

        stream.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = stream.read()

        # frame preprocessing
        if (args.rotate):
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if args.crop:
            if roi is False:
                cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
                roi = cv2.selectROI("Select ROI", frame)
                cv2.destroyWindow("Select ROI")

            frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]

        # apply EasyOCR to the frame
        result = reader.readtext(frame, batch_size=4)
        text, prob = "", 0

        if result:
            best_result = max(result, key=lambda item: item[2])
            bbox, text, prob = best_result[0], filter_text(best_result[1], args.whitelist), best_result[2]
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 1)

        # Define the timing and length for each subtitle object.
        start_time_ms = stream.get(cv2.CAP_PROP_POS_MSEC)
        end_time_ms = start_time_ms + (tDelta * 1000)

        # Append data to dict
        entries['frame_number'].append(frame_count)
        entries['timecode_ms'].append(start_time_ms)
        entries['ocr_text'].append(str(text).strip())
        entries['ocr_confidence'].append(prob)

        # # Append data to json_output list
        if args.json:
            json_output.append({
                'frame_number': frame_count,
                'timecode_ms': start_time_ms,
                'ocr_text': text,
                'ocr_confidence': prob  # The confidence of the OCR string
            })

        # Display a video preview with bounding boxes if the preview is enabled.
        if args.preview:
            cv2.imshow("Preview", frame)

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
            'ocr_language': args.language,
            'analysis_frame_interval': args.frame_rate,
            'character_whitelist': args.whitelist
        }

        # Insert the extra_info dictionary at the beginning of the json_output list
        json_output.insert(0, extra_info)

        # Define the output filename for the JSON file.
        output_json_filename = os.path.join(outputFolderPath, videoName + '_OCR.json')
        print(f"Preparing to write JSON to file: {output_json_filename}")

        # Try to write JSON data to the file.
        try:
            with open(output_json_filename, 'w') as json_file:
                json.dump(json_output, json_file, indent=4)
            print("JSON file written successfully")
        except Exception as e:
            print(f"Error while writing JSON file: {e}")

    # Write dataframe to csv
    output_csv_filename = os.path.join(outputFolderPath, videoName + '_OCR.csv')
    print(f"Preparing to write CSV to file: {output_csv_filename}")
    pd.DataFrame.from_dict(entries).to_csv(output_csv_filename, index=True)
    print("CSV file written successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract text from video using OCR and generate SRT file')
    parser.add_argument('-v', '--video', help='Path to the video file', required=True)
    parser.add_argument('-o', '--output', help='Path to the output folder')
    parser.add_argument('-l', '--language', help='Language for OCR', default='en')
    parser.add_argument('-f', '--frame_rate', help='Number of frames to skip for processing', type=int, default=10)
    parser.add_argument('-p', '--preview', help='Enable preview of the video', action='store_true')
    parser.add_argument('-w', '--whitelist', help='whitelist characters to improve OCR result',
                        default='1234567890-')
    parser.add_argument('-j', '--json', help='Enable JSON output', action='store_true')
    parser.add_argument('-c', '--crop',
                        help='Crop the video frame by dragging selected region with cursor', action='store_true')
    parser.add_argument('-r', "--rotate", help="Rotate the video frame 90 degrees clockwise", action='store_true')

    args = parser.parse_args()

    main(args)
