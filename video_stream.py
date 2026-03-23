import cv2
import imutils
import numpy as np
import pytesseract
import re

# pytesseract.pytesseract.tesseract_cmd = r"/opt/homebrew/bin/tesseract"

cap = cv2.VideoCapture(0)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    cnts, _ = cv2.findContours(edged.copy(),
                               cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    plate = None
    screenCnt = None
    best_conf = 0

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h
            if 3.0 < aspect < 8.0:
                screenCnt = approx
                # confidence: cât de aproape este de aspectul ideal 4.5
                best_conf = max(0, 100 - abs((aspect - 4.5) * 30))
                break

    if screenCnt is not None:
        pts = screenCnt.reshape(4, 2)
        cv2.polylines(frame, [screenCnt], True, (0,165,255), 3)

        warped = four_point_transform(frame, pts)

        plate_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)

        thresh = cv2.adaptiveThreshold(plate_gray, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       31, 15)

        config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7'
        text = pytesseract.image_to_string(thresh, config=config)
        text = re.sub(r'[^A-Z0-9]', '', text)

        # afișăm PROCENT + TEXT lângă chenar
        if len(text) > 0:
            # poziționare text lângă colțul de sus-stânga al chenarului
            tx, ty = int(pts[0][0]), int(pts[0][1]) - 15
            if ty < 30:
                ty = int(pts[3][1]) + 40  # dacă e sus în cadru, mutăm dedesubt

            cv2.putText(frame, f"{text}  |  {best_conf:.1f}%",
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,165,255), 3)

        cv2.imshow("Warped Plate", warped)
        cv2.imshow("Thresh", thresh)

    cv2.imshow("Edges", edged)
    cv2.imshow("Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
