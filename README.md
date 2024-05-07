# Two-Wheeler Traffic Rule Violation Detection
This project is aimed at detecting and identifying traffic rule violations by two-wheeler riders, with a focus on the detection of [violations](https://github.com/pratham-jaiswal/two-wheeler-traffic-rule-violation/edit/main/README.md#detected-objects). The project utilizes three Models trained on Roboflow Cloud <sup>[[1]](https://universe.roboflow.com/nckh-2023/helmet-detection-project/model/13)[[2]](https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i/model/21)[[3]](https://universe.roboflow.com/prathamjaiswal/two-wheeler-lane-detection/model/3)</sup> to detect [objects of interest](https://github.com/pratham-jaiswal/two-wheeler-traffic-rule-violation/edit/main/README.md#detected-objects) in video frames. It saves the images of two wheelers which violated the traffic rules, along with image of their license plates.

## Dataset
The datasets used for this project are [Helmet Detection Project](https://universe.roboflow.com/nckh-2023/helmet-detection-project), [Face Detection](https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i), and [Two Wheeler Lane Detection](https://universe.roboflow.com/prathamjaiswal/two-wheeler-lane-detection).

> The Two Wheeler Lane Detection Dataset is created by [Pratham Jaiswal (me)](https://github.com/pratham-jaiswal), [Arnav Rawat](https://github.com/ArnavRw21), and [Shubham Sharma](https://github.com/Shubham1709).

## Detected Objects
- Helmet Detection Project: Two-wheeler/motorcyclist, Helmet, License Plate
- Face Detection: Human Face
- Two Wheeler Lane Detection: Front-facing motorcycle, Rear-facing motorcycle

## Violations
- Wrong Lane: Driving away from the camera.
- No Helmet: Any rider not wearing a helmet.
- Triple riding: More than two riders.

## Process Flow
1. **Motorcycle Detection:**
   - Detects all two-wheelers/motorcycles in a frame.
2. **Bounding Box Extraction:**
   - For each detected motorcycle, extracts its bounding box.
3. **Orientation Check:**
   - Determines if the motorcycle is front-facing or rear-facing.
   - Flags a "Wrong Lane Violation" if the motorcycle is rear-facing.
4. **Face and Helmet Detection:**
   - Detects faces and helmets within the cropped image.
   - Counts the number of faces.
   - Reduces the face count if the detected face and helmet areas overlap by more than 60%.
5. **No Helmet Violation:**
   - Detects helmets again and counts them.
   - Flags a "No Helmet Violation" if no helmets are detected or if the number of faces is greater than 1.
6. **Triple Riding Violation:**
   - Sums up the final counts of helmets and faces.
   - Flags a "Triple Riding Violation" if the sum is greater than 2.
7. **License Plate Detection:**
   - If any violation is detected, captures the license plate using the [OCR.Space API](https://ocr.space/OCRAPI).
8. **Saving Violation Data:**
   - Saves the violated motorcycle image along with its license plate image and text.
   - Records the list of violations for each image.

## Getting Started
- Clone this repository,
    ```bash
    git clone https://github.com/pratham-jaiswal/two-wheeler-traffic-rule-violation.git
    ```
- Get your OCR.Space API key from [here](https://ocr.space/OCRAPI).
- Get your Roboflow API key by following this [guide](https://docs.roboflow.com/api-reference/authentication).
- Create a .env file with the following environment variables
    ```env
    OCR_SPACE_API=YOUR_OCRSPACE_API_KEY
    ROBOFLOW_API_KEY=YOUR_ROBOFLOW_API_KEY  
    ```
- Put the video in your directory, which contains ***main.py***, with the name ***input.mp4***.
- Run the ***main.py***.

## License
This project is provided under the [MIT License](https://github.com/pratham-jaiswal/two-wheeler-traffic-rule-violation/blob/main/LICENSE).
