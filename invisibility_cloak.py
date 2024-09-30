import cv2
import numpy as np
import time

def create_background(cap, num_frames=30):
    print("Capturing background. Please move out of frame.")
    backgrounds = []
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            backgrounds.append(frame)
        else:
            print(f"Warning: Could not read frame {i+1}/{num_frames}")
        time.sleep(0.1)
    if backgrounds:
        return np.median(backgrounds, axis=0).astype(np.uint8)
    else:
        raise ValueError("Could not capture any frames for background")

def create_mask(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)
    return mask

def apply_cloak_effect(frame, mask, background):
    mask_inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    bg = cv2.bitwise_and(background, background, mask=mask)
    return cv2.add(fg, bg)

def main():
    print("OpenCV version:", cv2.__version__)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Please move out of the frame in 5 seconds...")
    time.sleep(5)

    try:
        background = create_background(cap)
    except ValueError as e:
        print(f"Error: {e}")
        cap.release()
        return

    # Uncomment the color you want to use for the cloak

    # Red cloak (Two ranges due to red spanning the HSV spectrum)
    # lower_color1 = np.array([0, 120, 70])
    # upper_color1 = np.array([10, 255, 255])
    # lower_color2 = np.array([170, 120, 70])
    # upper_color2 = np.array([180, 255, 255])

    # Green cloak
    # lower_color = np.array([35, 50, 50])
    # upper_color = np.array([85, 255, 255])

    # Blue cloak
    lower_color = np.array([90, 50, 50])
    upper_color = np.array([130, 255, 255])

    # Yellow cloak
    # lower_color = np.array([20, 100, 100])
    # upper_color = np.array([30, 255, 255])

    # Cyan cloak
    # lower_color = np.array([80, 100, 100])
    # upper_color = np.array([90, 255, 255])

    # Magenta cloak
    # lower_color = np.array([140, 50, 50])
    # upper_color = np.array([170, 255, 255])

    # Orange cloak
    # lower_color = np.array([10, 100, 100])
    # upper_color = np.array([25, 255, 255])

    # White cloak
    # lower_color = np.array([0, 0, 200])
    # upper_color = np.array([180, 20, 255])

    # Black cloak
    # lower_color = np.array([0, 0, 0])
    # upper_color = np.array([180, 255, 30])

    # Gray cloak
    # lower_color = np.array([0, 0, 40])
    # upper_color = np.array([180, 255, 200])

    # Brown cloak
    # lower_color = np.array([10, 100, 20])
    # upper_color = np.array([20, 255, 200])

    print("Starting main loop. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            time.sleep(1)
            continue

        # For red cloak, we need to apply two masks (due to HSV spectrum wrapping)
        # Uncomment this if using red cloak
        # mask1 = create_mask(frame, lower_color1, upper_color1)
        # mask2 = create_mask(frame, lower_color2, upper_color2)
        # mask = mask1 + mask2

        # For other colors, use a single mask
        mask = create_mask(frame, lower_color, upper_color)

        result = apply_cloak_effect(frame, mask, background)

        cv2.imshow('Invisible Cloak', result)

        if cv2.getWindowProperty('Invisible Cloak', cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
