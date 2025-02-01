import os
from PIL import Image, ImageDraw
import cv2
from datetime import datetime, timezone, timedelta
import requests
import json
import re
from tqdm import tqdm
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from super_image import DrlnModel, ImageLoader

import warnings
warnings.filterwarnings('ignore')

load_dotenv()

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("ROBOFLOW_API_KEY")
_set_if_undefined("OCR_SPACE_API")

def upscale_image(image_path, scale):
    image = Image.open(image_path)

    image_name = os.path.splitext(image_path)[0]

    model = DrlnModel.from_pretrained('eugenesiow/drln', scale=scale)
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)

    ImageLoader.save_image(preds, f'{image_name}_{scale}x.png')

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
    if len(data["ParsedResults"]) == 0:
        return ""
    lpnum = data["ParsedResults"][0]["ParsedText"].replace('\n', '').replace('\\n', '').replace(' ', '')

    return lpnum

def draw_detections(p1, p2, p3, img):
    class_colors = {
        'helmet': 'blue',
        'motorcyclist': 'green',
        'license_plate': 'red',
        'face': 'darkmagenta',
        'front': 'darkgoldenrod',
        'rear': 'darkorchid'
    }
    
    draw = ImageDraw.Draw(img)

    preds = {'predictions': p1['predictions'] + p2['predictions'] + p3['predictions']}

    for prediction in preds['predictions']:
        x, y, width, height = (
            prediction['x'],
            prediction['y'],
            prediction['width'],
            prediction['height']
        )
        
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2
        
        class_name = prediction['class']
        confidence = prediction['confidence']
        
        label_color = class_colors.get(class_name, 'black')

        if class_name=='motorcyclist':
            draw.rectangle([x1, y1, x2, y1+14], fill=label_color)
            label_position = (x1 + 5, y1 + 2)
        else:
            draw.rectangle([x1, y1-14, x2, y1], fill=label_color)
            label_position = (x1 + 5, y1-12)
            
        draw.rectangle([x1, y1, x2, y2], outline=label_color, width=2)

        label = f"{class_name} ({confidence:.2f})"
        draw.text(label_position,label, fill='white')

    return img

# Roboflow API keys
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

custom_configuration = InferenceConfiguration(confidence_threshold=0.4, iou_threshold=0.4)
client1 = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=roboflow_api_key
)
client1.configure(custom_configuration)

custom_configuration = InferenceConfiguration(confidence_threshold=0.4, iou_threshold=0.3)
client2 = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=roboflow_api_key
)
client2.configure(custom_configuration)

custom_configuration = InferenceConfiguration(confidence_threshold=0.1, iou_threshold=0.1)
client3 = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=roboflow_api_key
)
client3.configure(custom_configuration)

video_path = 'input.mp4'
cap = cv2.VideoCapture(video_path)

# Get video details
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(5)
total_frames = int(cap.get(7))

# Violate Date folder
current_date = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%d-%m-%Y")
folder_path = os.path.join(os.getcwd(), f"Violations/{current_date}")
os.makedirs(folder_path, exist_ok=True)

# Process every 60th frame
for frame_number in tqdm(range(0, total_frames, 180), desc="Processing frames", unit="frames"):
    ret, frame = cap.read()
    if not ret:
        break

    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    image_path = "temp_frame.jpg"
    pil_frame.save(image_path)

    ## Can extremely slow down the process as the bigger the image the more resources it needs to be upscaled.
    # upscale_image(image_path, 2)
    # image_path = "temp_frame_4x.png"
    # pil_frame = Image.open(image_path)

    r1 = client1.infer(image_path, model_id="helmet-detection-project/13")
    pred1 = r1['predictions']

    for pr1 in pred1:
        helmet_detected = False
        face_detected = False
        rear_detected = False
        more_than_two_detected = False
        num_faces_detected = 0
        num_helmets_detected = 0

        if pr1['class'] == 'motorcyclist':
            motorcyclist_x, motorcyclist_y, motorcyclist_width, motorcyclist_height = pr1['x'], pr1['y'], pr1['width'], pr1['height']
            
            motorcyclist_x1, motorcyclist_y1 = int(motorcyclist_x - motorcyclist_width / 2), int(motorcyclist_y - motorcyclist_height / 2)
            motorcyclist_x2, motorcyclist_y2 = int(motorcyclist_x + motorcyclist_width / 2), int(motorcyclist_y + motorcyclist_height / 2)
            
            motorcyclist_image = pil_frame.crop((motorcyclist_x1, motorcyclist_y1, motorcyclist_x2, motorcyclist_y2))
            motorcyclist_image.save("temp_motorcyclist_image.jpg")

            # Lane check
            r3 = client3.infer("temp_motorcyclist_image.jpg", model_id="two-wheeler-lane-detection/3")
            lane = r3

            if lane['predictions']:
                max_conf = max(lane['predictions'], key=lambda x: x['confidence'])
                lane['predictions'] = [max_conf]
            
            pred3 = lane['predictions']
            
            for lane_prediction in pred3:
                if lane_prediction['class'] == 'rear':
                    rear_x, rear_y, rear_width, rear_height = lane_prediction['x'], lane_prediction['y'], lane_prediction['width'], lane_prediction['height']
                    rear_detected = True

            # Face detected
            r2 = client2.infer("temp_motorcyclist_image.jpg", model_id="face-detection-mik1i/21")
            pred2 = r2['predictions']

            for face_prediction in pred2:
                if face_prediction['class'] == 'face':
                    face_x, face_y, face_width, face_height = face_prediction['x'], face_prediction['y'], face_prediction['width'], face_prediction['height']

                    if motorcyclist_x1 < face_x < motorcyclist_x2 and motorcyclist_y1 < face_y < motorcyclist_y2:
                        num_faces_detected += 1

                        # Avoid detecting helmet and face in same area and calculating number of people incorrectly
                        for helmet_prediction in pred1:
                            if helmet_prediction['class'] == 'helmet':
                                helmet_x, helmet_y, helmet_width, helmet_height = helmet_prediction['x'], helmet_prediction['y'], helmet_prediction['width'], helmet_prediction['height']
                                
                                face_x1 = face_x - face_width / 2
                                face_y1 = face_y - face_height / 2
                                face_x2 = face_x + face_width / 2
                                face_y2 = face_y + face_height / 2

                                helmet_x1 = helmet_x - helmet_width / 2
                                helmet_y1 = helmet_y - helmet_height / 2
                                helmet_x2 = helmet_x + helmet_width / 2
                                helmet_y2 = helmet_y + helmet_height / 2

                                overlap_x1 = max(face_x, helmet_x)
                                overlap_y1 = max(face_y, helmet_y)
                                overlap_x2 = min(face_x + face_width, helmet_x + helmet_width)
                                overlap_y2 = min(face_y + face_height, helmet_y + helmet_height)

                                overlap_width = max(0, overlap_x2 - overlap_x1)
                                overlap_height = max(0, overlap_y2 - overlap_y1)

                                overlap_area = overlap_width * overlap_height

                                face_area = face_width * face_height

                                if overlap_area / face_area > 0.6:
                                    num_faces_detected -= 1
                                    break

            if num_faces_detected > 0:
                face_detected = True

            # Helmet check
            for helmet_prediction in pred1:
                if helmet_prediction['class'] == 'helmet':
                    helmet_x, helmet_y, helmet_width, helmet_height = helmet_prediction['x'], helmet_prediction['y'], helmet_prediction['width'], helmet_prediction['height']

                    if motorcyclist_x1 < helmet_x < motorcyclist_x2 and motorcyclist_y1 < helmet_y < motorcyclist_y2:
                        helmet_detected = True
                        num_helmets_detected += 1

            # More than two riding
            if num_faces_detected + num_helmets_detected > 2:
                more_than_two_detected = True

            # r4 = m1.predict("temp_motorcyclist_image.jpg", confidence=60, overlap=40)
            r4 = client1.infer("temp_motorcyclist_image.jpg", model_id="helmet-detection-project/13")
            colored_motorcycle = draw_detections(r4, r2, lane, motorcyclist_image)
            
            # Violated license plate
            if not helmet_detected or face_detected or rear_detected or more_than_two_detected:
                
                violation_names = []
                if not helmet_detected or face_detected:
                    violation_names.append('no_helmet')
                if rear_detected:
                    violation_names.append('wrong_lane')
                if more_than_two_detected:
                    violation_names.append('triple_riding')

                timestamp = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%d-%m-%Y %H %M %S")
                image_name = ", ".join(violation_names) + f" - {timestamp}"
                lp_detected = False

                for pr11 in r4["predictions"]:
                    if pr11['class'] == 'license_plate':
                        license_plate_x, license_plate_y, license_plate_width, license_plate_height = pr11['x'], pr11['y'], pr11['width'], pr11['height']
                        license_plate_x1, license_plate_y1 = int(license_plate_x - license_plate_width / 2), int(license_plate_y - license_plate_height / 2)
                        license_plate_x2, license_plate_y2 = int(license_plate_x + license_plate_width / 2), int(license_plate_y + license_plate_height / 2)
        
                        license_plate_image = motorcyclist_image.crop((license_plate_x1, license_plate_y1, license_plate_x2, license_plate_y2))
                        
                        license_plate_image.save("temp_lp.jpg")

                        upscale_image("temp_lp.jpg", 4)
                        license_plate_image = Image.open("temp_lp_4x.png")
                        
                        lpnum = ocr_space_file(filename="temp_lp_4x.png", overlay=False, api_key=os.getenv("OCR_SPACE_API"), language='eng')   
                        
                        if lpnum.strip():
                            image_name = lpnum + " - " + image_name
                        else:
                            image_name = image_name
                        image_folder_path = os.path.join(folder_path, image_name)
                        os.makedirs(image_folder_path, exist_ok=True)

                        violated_motorcycle_image_path = os.path.join(image_folder_path, f"{lpnum} - motorcyclist.jpg")
                        colored_motorcycle.save(violated_motorcycle_image_path)

                        violated_motorcycle_lp_image_path = os.path.join(image_folder_path, f"{lpnum} - license_plate.jpg")
                        license_plate_image.save(violated_motorcycle_lp_image_path)

                        lp_text_path = os.path.join(image_folder_path, f"{lpnum} - license_plate_number.txt")
                        with open(lp_text_path, 'w') as file:
                            file.write(f"Violated License Plate Number - {lpnum}")

                        lp_detected = True

                        if os.path.exists("temp_lp.jpg"):
                            os.remove("temp_lp.jpg")
                        if os.path.exists("temp_lp_4x.png"):
                            os.remove("temp_lp_4x.png")
                        break

                if not lp_detected:
                    image_folder_path = os.path.join(folder_path, image_name)
                    os.makedirs(image_folder_path, exist_ok=True)
                    violated_motorcycle_image_path = os.path.join(image_folder_path, f"motorcyclist.jpg")

                    colored_motorcycle.save(violated_motorcycle_image_path)

if os.path.exists("temp_motorcyclist_image.jpg"):
    os.remove("temp_motorcyclist_image.jpg")

if os.path.exists("temp_frame.jpg"):
    os.remove("temp_frame.jpg")

if os.path.exists("temp_frame_4x.png"):
    os.remove("temp_frame_4x.png")

cap.release()

print("Video processing completed.")