import cv2
import torch
import numpy as np
import os
import requests
import json
import re

def ocr_space_file(filename, overlay, api_key, language):

    payload = {
                'isOverlayRequired': overlay,
                'apikey': api_key,
                'language': language,
                'OCREngine': 2,
            }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    data = json.loads(r.content.decode())

    lines = data["ParsedResults"][0]["TextOverlay"]["Lines"]
    lpnum = "".join(line["LineText"] for line in lines)
    lpnum = re.sub(r'[^a-zA-Z0-9]', '', lpnum)
    


path = './models/512_2_5m.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

output_dir = "nohelmet lp"
os.makedirs(output_dir, exist_ok=True)

image_path = './Helmet Detection Dataset/test/images/IMG_5076_MOV-38_jpg.rf.7e5eb3bbcc477688344bbbdce2d2da5d.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (1020, 600))

results = model(image)

frame = np.squeeze(results.render()[0])

detections = results.pred[0]

twowheelers = detections[(detections[:, 5] == 2) & (detections[:, 4] > 0.75)]

count = 0

for tw in twowheelers:
    count += 1
    print(1)
    x1, y1, x2, y2, conf, class_id = tw.tolist()

    helmets = detections[(detections[:, 5] == 0) & (detections[:, 4] > 0.5)]
    helmet_detected = False
    for helmet in helmets:
        print(2)
        hx1, hy1, hx2, hy2, hconf, hclass_id = helmet.tolist()
        if x1 < hx1 < x2 and y1 < hy1 < y2:
            helmet_detected = True
            print(3)
            break

    if not helmet_detected:
        print(4)
        license_plates = detections[detections[:, 5] == 1]
        for license_plate in license_plates:
            print(5)
            lx1, ly1, lx2, ly2, lconf, lclass_id = license_plate.tolist()
            if x1 < lx1 < x2 and y1 < ly1 < y2:
                if conf > 0.75:
                    print(6)
                    tw_image = frame[int(y1):int(y2), int(x1):int(x2)]
                    output_path = os.path.join(output_dir, f"{count}_two_wheeler.jpg")
                    cv2.imwrite(output_path, tw_image)
                    lp_image = frame[int(ly1):int(ly2), int(lx1):int(lx2)]
                    output_path = os.path.join(output_dir, f"{count}_license_plate.jpg")
                    cv2.imwrite(output_path, lp_image)

    frame = np.squeeze(results.render()[0])
    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break



cv2.imshow("Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# cap = cv2.VideoCapture('./output.mp4')
# count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     count += 1
#     if count % 10 != 0:
#         continue

#     frame = cv2.resize(frame, (1020, 600))
#     results = model(frame)

#     detections = results.pred[0]

#     twowheelers = detections[(detections[:, 5] == 2) & (detections[:, 4] > 0.75)]

#     for tw in twowheelers:
#         x1, y1, x2, y2, conf, class_id = tw.tolist()

#         helmets = detections[(detections[:, 5] == 0) & (detections[:, 4] > 0.5)]
#         helmet_detected = False
#         for helmet in helmets:
#             hx1, hy1, hx2, hy2, hconf, hclass_id = helmet.tolist()
#             if x1 < hx1 < x2 and y1 < hy1 < y2:
#                 helmet_detected = True
#                 break

#         if not helmet_detected:
#             license_plates = detections[detections[:, 5] == 1]
#             for license_plate in license_plates:
#                 lx1, ly1, lx2, ly2, lconf, lclass_id = license_plate.tolist()
#                 if x1 < lx1 < x2 and y1 < ly1 < y2:
#                     if conf > 0.90:
#                         license_plate_image = frame[int(y1):int(y2), int(x1):int(x2)]
#                         output_path = os.path.join(output_dir, f"license_plate_{count}.jpg")
#                         cv2.imwrite(output_path, license_plate_image)

#     frame = np.squeeze(results.render()[0])
#     cv2.imshow("Video", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()