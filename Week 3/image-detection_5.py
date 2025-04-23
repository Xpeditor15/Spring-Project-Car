import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2

# Paths
model_path = "/home/pi/Downloads/Week_3/imageDetection/model_unquant.tflite"
label_path = "/home/pi/Downloads/Week_3/imageDetection/labels.txt"

# Load labels
with open(label_path, 'r') as f:
    symbol_names = [line.strip() for line in f.readlines()]

# Load model
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

while True:
    # Capture frame
    frame = picam2.capture_array()

    # Preprocess
    input_frame = cv2.resize(frame, (224, 224))
    input_frame = input_frame.astype(np.float32) / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_frame)
    interpreter.invoke()

    # Get output (class probabilities)
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_class = np.argmax(output)
    confidence = output[predicted_class]

    # Display result
    if confidence > 0.5:  # Confidence threshold
        label = f"{symbol_names[predicted_class]}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Show frame
    cv2.imshow('Symbol Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
picam2.stop()
cv2.destroyAllWindows()