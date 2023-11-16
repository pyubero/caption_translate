import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pytesseract import pytesseract


def export_srt(filename, start, end, captions):
    assert len(start) == len(captions)
    assert len(end) == len(captions)
    ncaption = -1

    with open(filename, "w+") as file:
        for ii, caption in enumerate(captions):
            if caption=="":
                continue

            ncaption += 1
            ini_caption = convert_seconds(start[ii]/FPS)
            fin_caption = convert_seconds(end[ii]/FPS)

            _ = file.write(f"{ncaption}\n")
            _ = file.write(f"{ini_caption} --> {fin_caption}\n")
            _ = file.write(caption+"\n")


def convert_seconds(sec):
   sec = sec % (24 * 3600)
   hour = sec // 3600
   sec %= 3600
   min = sec // 60
   sec %= 60
   return "%02d:%02d:%02.3f" % (hour, min, sec)


# --- Tunable parameters ---
VIDEO = "video.mp4"
LANG_IN = "deu"
OCR = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
THRESHOLD = 5_000
SATURATION_THRESHOLD = 200
VALUE_THRESHOLD = 200

# Initialize
# ... pytesseract
pytesseract.tesseract_cmd = OCR
options = f"-l {LANG_IN}"

# ... video file
cap = cv2.VideoCapture(VIDEO)
FPS = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# ... captions and counters
tStart = datetime.now()
captions = ["",]
start, end = [0,], []
kernel = np.ones((3, 3), np.uint8)
erosion_old = 255 * np.ones((HEIGHT, WIDTH), dtype="uint8")
tStart = datetime.now()

for iframe in range(total_frames):
    ret, frame = cap.read()

    # PROCESS IMAGE
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    saturation_mask = 255 * (frame[:, :, 1] > SATURATION_THRESHOLD)
    captions_mask = 255 * (frame[:, :, 2] > VALUE_THRESHOLD)
    processed = cv2.bitwise_and(saturation_mask,captions_mask)

    ret, thresh = cv2.threshold(
        processed.astype("uint8"),
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    erosion = cv2.erode(thresh, kernel, iterations = 1)

    # If eroded image is sufficiently different (to speed up)...
    if np.sum(erosion_old != erosion) > THRESHOLD:
        # DETECT TEXT
        text = pytesseract.image_to_string(erosion, config=options)

        end.append(iframe)
        start.append(iframe)
        captions.append(text)

        fps = iframe / (datetime.now()-tStart).total_seconds()
        print(f"[{iframe/total_frames:.2%}] [{fps:.0f} fps] {text}")

    # Prepare next iteration
    erosion_old = erosion


end.append(iframe)
cv2.destroyAllWindows()
cap.release()

fps = iframe / (datetime.now()-tStart).total_seconds()
print(f"\nFinished processing at {fps:.1f} fps.")

# Export captions in original language
videoname = ".".join(VIDEO.split(".")[:-1])
outputfilename = f"{videoname}_{LANG_IN}.srt"
export_srt(outputfilename, start, end, captions)
print(f"\nCaptions exported in {LANG_IN}.")
