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
# FRAMES_DIR = ROOT / "frames"
# st.write(FRAMES_DIR)
# FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# # Load Haarcascade for face detection
# CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# # NOTE: The callback will be called in another thread,
# #       so use a queue here for thread-safety to pass the data
# #       from inside to outside the callback.
# result_queue: "queue.Queue[List[str]]" = queue.Queue()

# # Counter for saved frames
# saved_frame_count = 0
# MAX_FRAMES = 20

# def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
#     global saved_frame_count
#     if saved_frame_count >= MAX_FRAMES:
#         return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")

#     image = frame.to_ndarray(format="bgr24")

#     # Convert to grayscale for face detection
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         if saved_frame_count >= MAX_FRAMES:
#             break

#         # Extract the face region
#         face = image[y:y+h, x:x+w]

#         # Save the face frame
#         frame_name = f"face_{frame.pts}_{x}_{y}.jpg"
#         frame_path = FRAMES_DIR / frame_name
#         cv2.imwrite(str(frame_path), face)

#         # Increment the saved frame count
#         saved_frame_count += 1

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
# def list_saved_files():
#     files = list(FRAMES_DIR.glob("*.jpg"))
#     return [str(file.name) for file in files]
    
# if st.checkbox("Show saved face frame names", value=True):
#     if webrtc_ctx.state.playing:
#         labels_placeholder = st.empty()
#         while True:
#             frame_name = result_queue.get()
#             labels_placeholder.text(f"Saved: {frame_name}")
            
# if st.button("List all saved files"):
#     saved_files = list_saved_files()
#     st.write("Saved Files:")
#     st.write(saved_files)
import logging
import queue
from pathlib import Path
from typing import List

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers

with st.sidebar:
    # st.image("./assets/faceman_cropped.png", width=260)

    title = '<p style="font-size: 25px;font-weight: 550;">Face Detection Settings</p>'
    st.markdown(title, unsafe_allow_html=True)

    # choose the mode for detection
    mode = st.radio("Choose Face Detection Mode", ('Home','Webcam Image Capture','Webcam Realtime Attendance Fill','Train Faces','Manual Attendance'), index=0)
    if mode == "Home":
        detection_mode = mode
    if mode == "Webcam Image Capture":
        detection_mode = mode
    elif mode == 'Webcam Realtime Attendance Fill':
        detection_mode = mode
    elif mode == 'Train Faces':
        detection_mode = mode
    elif mode == 'Train with Quantum Image edge Detection':
        detection_mode = mode
    elif mode == 'Manual Attendance':
        detection_mode = mode

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# Create a directory to save frames
FRAMES_DIR = ROOT / "detected_faces"
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

        if not user_name or not roll_number:
            st.error("Please enter both name and roll number to save frames.")
            break

        # Extract the face region
        face = image[y:y+h, x:x+w]

        # Save the face frame
        frame_name = f"{user_name}.{roll_number}.{saved_frame_count + 1}.jpg"
        frame_path = FRAMES_DIR / frame_name
        cv2.imwrite(str(frame_path), face)

        # Increment the saved frame count
        saved_frame_count += 1

        # Draw a rectangle around the detected face (for visualization)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Add the saved face frame path to the result queue for display
        result_queue.put(frame_name)

    return av.VideoFrame.from_ndarray(image, format="bgr24")
if detection_mode == "Webcam Image Capture":
    # User inputs for name and roll number
    user_name = st.text_input("Enter your name:")
    roll_number = st.text_input("Enter your roll number:")
    
    webrtc_ctx = webrtc_streamer(
        key="save-face-frames",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": get_ice_servers()},
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    def list_saved_files():
        files = list(FRAMES_DIR.glob("*.jpg"))
        return [str(file.name) for file in files]
    
    if st.checkbox("Show saved face frame names", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            while True:
                frame_name = result_queue.get()
                labels_placeholder.text(f"Saved: {frame_name}")
    
    if st.button("List all saved files"):
        saved_files = list_saved_files()
        st.write("Saved Files:")
        st.write(saved_files)


if detection_mode == "Train Faces":
        
    # Path for face image database
    path = FRAMES_DIR

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    st.text("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('TrainingImageLabel/Trainner.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    st.text("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
