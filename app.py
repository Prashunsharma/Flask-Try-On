import os
import cv2
import cvzone
from flask import Flask, render_template, request, jsonify, Response
from cvzone.PoseModule import PoseDetector

# Initialize Flask app
app = Flask(__name__)

# Initialize paths
original_folder = "shirt"
processed_folder = "processed_shirts"
os.makedirs(processed_folder, exist_ok=True)

# Calibration factor (pixels per cm) and size reduction
calibration_factor = 9  # Adjust based on your setup
size_reduction_factor = 0.75  # Scaling factor to reduce shirt size (e.g., 90%)

# Preprocessing Function with Size Reduction
def preprocess_shirts(shoulder_width_cm, torso_height_cm):
    """Preprocess shirt images to match user-provided dimensions with scaling."""
    shoulder_width_px = int(shoulder_width_cm * calibration_factor * size_reduction_factor)
    torso_height_px = int(torso_height_cm * calibration_factor * size_reduction_factor)

    shirt_files = [f for f in os.listdir(original_folder) if f.endswith(('.png', '.jpg'))]
    for shirt_file in shirt_files:
        shirt_path = os.path.join(original_folder, shirt_file)
        shirt_image = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
        
        if shirt_image is None:
            print(f"Error loading {shirt_path}, skipping...")
            continue
        
        aspect_ratio = shirt_image.shape[0] / shirt_image.shape[1]
        target_width = shoulder_width_px
        target_height = int(torso_height_px * aspect_ratio)

        resized_shirt = cv2.resize(shirt_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        processed_path = os.path.join(processed_folder, shirt_file)
        cv2.imwrite(processed_path, resized_shirt)
        print(f"Processed and saved: {processed_path}")

    print("Preprocessing complete. All shirts have been resized based on user dimensions.")

# Initialize camera and pose detector
detector = PoseDetector()
shirt_files = [f for f in os.listdir(processed_folder) if f.endswith(('.png', '.jpg'))]
current_shirt_index = 0

def load_shirt(index):
    shirt_path = os.path.join(processed_folder, shirt_files[index])
    shirt_image = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
    if shirt_image is None:
        print(f"Error: Unable to load shirt image {shirt_path}.")
        exit()
    return shirt_image

# Load initial shirt
img_Shirt = load_shirt(current_shirt_index)

# Adjustments for position
vertical_offset = -10
horizontal_offset = -10

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_shirts', methods=['POST'])
def process_shirts():
    shoulder_width_cm = float(request.form.get('shoulder_width'))
    torso_height_cm = float(request.form.get('torso_height'))
    
    preprocess_shirts(shoulder_width_cm, torso_height_cm)
    
    return jsonify({"message": "Shirts processed successfully!"})

@app.route('/virtual_fitting')
def virtual_fitting():
    def generate():
        global vertical_offset, horizontal_offset
        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            if not success:
                break

            img = cv2.resize(img, (960, 720))
            
            img = detector.findPose(img, draw=False)
            lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)
            
            if lmList:
                left_shoulder = lmList[11][:2]
                right_shoulder = lmList[12][:2]
                left_hip = lmList[23][:2]
                right_hip = lmList[24][:2]

                # Calculate shoulder center
                shoulder_center = [(left_shoulder[0] + right_shoulder[0]) // 2, 
                                   (left_shoulder[1] + right_shoulder[1]) // 2]

                # Use preprocessed shirt dimensions
                shirt_width = img_Shirt.shape[1]
                shirt_height = img_Shirt.shape[0]

                # Adjust shirt position (vertically upward)
                center_x = int(shoulder_center[0] + horizontal_offset)
                center_y = int(shoulder_center[1] - shirt_height * 0.6 + vertical_offset)

                # Overlay shirt
                try:
                    img = cvzone.overlayPNG(img, img_Shirt, [center_x - shirt_width // 2, center_y])
                except Exception as e:
                    print(f"Error overlaying shirt: {e}")
            
            _, img_encoded = cv2.imencode('.jpg', img)
            frame = img_encoded.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/next_shirt', methods=['GET'])
def next_shirt():
    global current_shirt_index, img_Shirt
    current_shirt_index = (current_shirt_index + 1) % len(shirt_files)
    img_Shirt = load_shirt(current_shirt_index)
    return jsonify({"message": "Switched to next shirt."})

@app.route('/previous_shirt', methods=['GET'])
def previous_shirt():
    global current_shirt_index, img_Shirt
    current_shirt_index = (current_shirt_index - 1) % len(shirt_files)
    img_Shirt = load_shirt(current_shirt_index)
    return jsonify({"message": "Switched to previous shirt."})

@app.route('/adjust_position', methods=['POST'])
def adjust_position():
    global vertical_offset, horizontal_offset
    direction = request.json.get('direction')
    
    if direction == 'w':
        vertical_offset -= 10
    elif direction == 's':
        vertical_offset += 10
    elif direction == 'a':
        horizontal_offset -= 10
    elif direction == 'd':
        horizontal_offset += 10
    
    return jsonify({
        "message": "Position adjusted.",
        "vertical_offset": vertical_offset,
        "horizontal_offset": horizontal_offset
    })

if __name__ == '__main__':
    app.run(debug=True)
