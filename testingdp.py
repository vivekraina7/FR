# import logging
# import queue
# from pathlib import Path
# from typing import List

# import av
# import cv2
# import numpy as np
# import streamlit as st
# from streamlit_webrtc import WebRtcMode, webrtc_streamer

# from download import download_file
# from turn import get_ice_servers

# HERE = Path(__file__).parent
# ROOT = HERE.parent

# logger = logging.getLogger(__name__)

# # Create a directory to save frames
# # D:\Viren\Qriocity\Realtime-Face-Detection\frames
# FRAMES_DIR = ROOT / "Realtime-Face-Detection" / "frames"
# st.write(FRAMES_DIR)
# FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# # NOTE: The callback will be called in another thread,
# #       so use a queue here for thread-safety to pass the data
# #       from inside to outside the callback.
# result_queue: "queue.Queue[List[str]]" = queue.Queue()


# def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
#     image = frame.to_ndarray(format="bgr24")

#     # Save the frame to the frames directory
#     frame_name = f"frame_{frame.pts}.jpg"
#     frame_path = FRAMES_DIR / frame_name
#     cv2.imwrite(str(frame_path), image)

#     # Add the saved frame path to the result queue for display
#     result_queue.put(frame_name)

#     return av.VideoFrame.from_ndarray(image, format="bgr24")

# webrtc_ctx = webrtc_streamer(
#     key="save-frames",
#     mode=WebRtcMode.SENDRECV,
#     rtc_configuration={"iceServers": get_ice_servers()},
#     video_frame_callback=video_frame_callback,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=True,
# )

# if st.checkbox("Show saved frame names", value=True):
#     if webrtc_ctx.state.playing:
#         labels_placeholder = st.empty()
#         while True:
#             frame_name = result_queue.get()
#             labels_placeholder.text(f"Saved: {frame_name}")



# import logging
# import queue
# from pathlib import Path
# from typing import List

# import av
# import cv2
# import numpy as np
# import streamlit as st
# from streamlit_webrtc import WebRtcMode, webrtc_streamer

# from download import download_file
# from turn import get_ice_servers

# # HERE = Path(__file__).parent
# # ROOT = HERE.parent
# HERE = Path(__file__).parent
# ROOT = HERE.parent
# logger = logging.getLogger(__name__)
# FRAMES_DIR = ROOT / "Realtime-Face-Detection" / "frames"
# st.write(FRAMES_DIR)
# # FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# # Load Haarcascade for face detection
# CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# # NOTE: The callback will be called in another thread,
# #       so use a queue here for thread-safety to pass the data
# #       from inside to outside the callback.
# result_queue: "queue.Queue[List[str]]" = queue.Queue()


# def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
#     image = frame.to_ndarray(format="bgr24")

#     # Convert to grayscale for face detection
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # Extract the face region
#         face = image[y:y+h, x:x+w]

#         # Save the face frame
#         frame_name = f"face_{frame.pts}_{x}_{y}.jpg"
#         frame_path = FRAMES_DIR / frame_name
#         cv2.imwrite(str(frame_path), face)

#         # Draw a rectangle around the detected face (for visualization)
#         cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

#         # Add the saved face frame path to the result queue for display
#         result_queue.put(frame_name)

#     return av.VideoFrame.from_ndarray(image, format="bgr24")

# webrtc_ctx = webrtc_streamer(
#     key="save-face-frames",
#     mode=WebRtcMode.SENDRECV,
#     rtc_configuration={"iceServers": get_ice_servers()},
#     video_frame_callback=video_frame_callback,
#     media_stream_constraints={"video": True, "audio": False},
#     async_processing=True,
# )

# if st.checkbox("Show saved face frame names", value=True):
#     if webrtc_ctx.state.playing:
#         labels_placeholder = st.empty()
#         while True:
#             frame_name = result_queue.get()
#             labels_placeholder.text(f"Saved: {frame_name}")



import logging
import queue
from pathlib import Path
from typing import List

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from download import download_file
from turn import get_ice_servers

HERE = Path(__file__).parent
ROOT = HERE.parent
logger = logging.getLogger(__name__)
FRAMES_DIR = ROOT / "Realtime-Face-Detection" / "frames"
st.write(FRAMES_DIR)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Load Haarcascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
result_queue: "queue.Queue[List[str]]" = queue.Queue()

# Counter for saved frames
saved_frame_count = 0
MAX_FRAMES = 20

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global saved_frame_count
    if saved_frame_count >= MAX_FRAMES:
        return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")

    image = frame.to_ndarray(format="bgr24")

    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        if saved_frame_count >= MAX_FRAMES:
            break

        # Extract the face region
        face = image[y:y+h, x:x+w]

        # Save the face frame
        frame_name = f"face_{frame.pts}_{x}_{y}.jpg"
        frame_path = FRAMES_DIR / frame_name
        cv2.imwrite(str(frame_path), face)

        # Increment the saved frame count
        saved_frame_count += 1

        # Draw a rectangle around the detected face (for visualization)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Add the saved face frame path to the result queue for display
        result_queue.put(frame_name)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="save-face-frames",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": get_ice_servers()},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.checkbox("Show saved face frame names", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            frame_name = result_queue.get()
            labels_placeholder.text(f"Saved: {frame_name}")
