import streamlit as st
import cv2 as cv
import mediapipe as mp
import json
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
from functions import position_data, calculate_distance, draw_line, overlay_image

# --- Page Config ---
st.set_page_config(page_title="Doctor Strange Filter", page_icon="🪄", layout="centered")

# --- Utility Functions ---
def limit_value(val: int, min_val: int, max_val: int) -> int:
    return max(min(val, max_val), min_val)

@st.cache_data
def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as file:
        return json.load(file)

config = load_config()

@st.cache_data
def load_images():
    inner_circle = cv.imread(config["overlay"]["inner_circle_path"], -1)
    outer_circle = cv.imread(config["overlay"]["outer_circle_path"], -1)
    if inner_circle is None or outer_circle is None:
        raise FileNotFoundError("Failed to load one or more overlay images.")
    return inner_circle, outer_circle

inner_circle, outer_circle = load_images()

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Video Processing Class ---
# We use a class so that each user connection gets its own instance of the 
# MediaPipe model and its own rotation degree state.
class DoctorStrangeProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.deg = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv.flip(img, 1)

        h, w, _ = img.shape
        rgb_frame = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                # Extract hand landmark positions
                (wrist, thumb_tip, index_mcp, index_tip,
                 middle_mcp, middle_tip, ring_tip, pinky_tip) = position_data(lm_list)

                index_wrist_distance = calculate_distance(wrist, index_mcp)
                index_pinky_distance = calculate_distance(index_tip, pinky_tip)
                
                if index_wrist_distance == 0:
                    continue
                    
                ratio = index_pinky_distance / index_wrist_distance

                if 0.5 < ratio < 1.3:
                    # Hand open (drawing lines)
                    fingers = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
                    for finger in fingers:
                        img = draw_line(img, wrist, finger,
                                          color=tuple(config["line_settings"]["color"]),
                                          thickness=config["line_settings"]["thickness"])
                    for i in range(len(fingers) - 1):
                        img = draw_line(img, fingers[i], fingers[i + 1],
                                          color=tuple(config["line_settings"]["color"]),
                                          thickness=config["line_settings"]["thickness"])

                elif ratio >= 1.3:
                    # Hand closed (drawing shields)
                    center_x, center_y = middle_mcp
                    diameter = round(index_wrist_distance * config["overlay"]["shield_size_multiplier"])

                    x1 = limit_value(center_x - diameter // 2, 0, w)
                    y1 = limit_value(center_y - diameter // 2, 0, h)
                    diameter = min(diameter, w - x1, h - y1)
                    
                    if diameter > 0:
                        self.deg = (self.deg + config["overlay"]["rotation_degree_increment"]) % 360
                        M1 = cv.getRotationMatrix2D((outer_circle.shape[1] // 2, outer_circle.shape[0] // 2), self.deg, 1.0)
                        M2 = cv.getRotationMatrix2D((inner_circle.shape[1] // 2, inner_circle.shape[0] // 2), -self.deg, 1.0)

                        rotated_outer = cv.warpAffine(outer_circle, M1, (outer_circle.shape[1], outer_circle.shape[0]))
                        rotated_inner = cv.warpAffine(inner_circle, M2, (inner_circle.shape[1], inner_circle.shape[0]))

                        img = overlay_image(rotated_outer, img, x1, y1, (diameter, diameter))
                        img = overlay_image(rotated_inner, img, x1, y1, (diameter, diameter))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI Layout ---
st.title("Doctor Strange Filter 🪄")
st.markdown("### Real-time Doctor Strange AR Filter")
st.markdown(
    "Using OpenCV, MediaPipe, and Streamlit WebRTC. "
    "Grant camera access, start the webcam, and show your hands to cast the spell!"
)

# Start WebRTC stream
webrtc_streamer(
    key="dr-strange",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=DoctorStrangeProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.markdown("---")
st.markdown("#### How to Use:")
st.markdown("- **Hand Open:** Draws mystic lines connecting your fingers.")
st.markdown("- **Hand Closed (Fist with Index/Pinky out):** Summons the rotating Doctor Strange shields.")
