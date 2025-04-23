import cv2
import numpy as np
from picamera2 import Picamera2
import time

# Initialize the Pi camera
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # Give camera time to initialize
    return picam2

# Mouse callback function to get HSV values
def get_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the HSV value of the pixel clicked
        hsv_frame = param["hsv_frame"]
        h, s, v = hsv_frame[y, x]
        color_name = param["color_name"]
        print(f"{color_name} HSV value at ({x},{y}): H={h}, S={s}, V={v}")
        
        # Add to the list of sampled values
        param["h_values"].append(h)
        param["s_values"].append(s)
        param["v_values"].append(v)
        
        # Calculate and display suggested range
        if len(param["h_values"]) > 0:
            h_min, h_max = max(0, min(param["h_values"]) - 10), min(180, max(param["h_values"]) + 10)
            s_min, s_max = max(0, min(param["s_values"]) - 30), min(255, max(param["s_values"]) + 30)
            v_min, v_max = max(0, min(param["v_values"]) - 30), min(255, max(param["v_values"]) + 30)
            
            print(f"Suggested range for {color_name}:")
            print(f"lower_{color_name} = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_{color_name} = np.array([{h_max}, {s_max}, {v_max}])")
            
            # Special case for red (which wraps around in HSV)
            if color_name == "red" and h_min < 10 and h_max > 170:
                print("Red color wraps around, consider using two ranges:")
                print(f"lower_red1 = np.array([0, {s_min}, {v_min}])")
                print(f"upper_red1 = np.array([10, {s_max}, {v_max}])")
                print(f"lower_red2 = np.array([170, {s_min}, {v_min}])")
                print(f"upper_red2 = np.array([180, {s_max}, {v_max}])")

def main():
    # Setup camera
    picam2 = setup_camera()
    
    # Create a window
    cv2.namedWindow('Camera Feed')
    
    # Initialize parameters for different colors
    color_params = {
        "red": {"h_values": [], "s_values": [], "v_values": [], "color_name": "red", "hsv_frame": None},
        "blue": {"h_values": [], "s_values": [], "v_values": [], "color_name": "blue", "hsv_frame": None},
        "green": {"h_values": [], "s_values": [], "v_values": [], "color_name": "green", "hsv_frame": None},
        "yellow": {"h_values": [], "s_values": [], "v_values": [], "color_name": "yellow", "hsv_frame": None},
        "black": {"h_values": [], "s_values": [], "v_values": [], "color_name": "black", "hsv_frame": None}
    }
    
    # Set the current color to sample (change this to the color you want to check)
    current_color = "red"
    cv2.setMouseCallback('Camera Feed', get_hsv_value, color_params[current_color])
    
    print(f"Sampling {current_color} color. Click on pixels of that color in the feed.")
    print("Press 'r' for red, 'g' for green, 'b' for blue, 'y' for yellow, 'k' for black")
    print("Press 'c' to clear samples for current color")
    print("Press 'q' to quit")
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Convert RGB to BGR and then to HSV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hsv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            
            # Update the HSV frame in the parameters
            color_params[current_color]["hsv_frame"] = hsv_frame
            
            # Display instructions
            cv2.putText(frame_bgr, f"Sampling: {current_color}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Camera Feed', frame_bgr)
            
            # Get keypresses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                current_color = "red"
                cv2.setMouseCallback('Camera Feed', get_hsv_value, color_params[current_color])
                print(f"Switched to sampling {current_color}")
            elif key == ord('g'):
                current_color = "green"
                cv2.setMouseCallback('Camera Feed', get_hsv_value, color_params[current_color])
                print(f"Switched to sampling {current_color}")
            elif key == ord('b'):
                current_color = "blue"
                cv2.setMouseCallback('Camera Feed', get_hsv_value, color_params[current_color])
                print(f"Switched to sampling {current_color}")
            elif key == ord('y'):
                current_color = "yellow"
                cv2.setMouseCallback('Camera Feed', get_hsv_value, color_params[current_color])
                print(f"Switched to sampling {current_color}")
            elif key == ord('k'):
                current_color = "black"
                cv2.setMouseCallback('Camera Feed', get_hsv_value, color_params[current_color])
                print(f"Switched to sampling {current_color}")
            elif key == ord('c'):
                # Clear samples for current color
                color_params[current_color]["h_values"] = []
                color_params[current_color]["s_values"] = []
                color_params[current_color]["v_values"] = []
                print(f"Cleared samples for {current_color}")
                
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        cv2.destroyAllWindows()
        print("Resources released")

if __name__ == "__main__":
    main() 