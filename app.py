# import streamlit as st
# import cv2
# import os
# from my_utils import alignment_procedure
# from mtcnn import MTCNN
# import glob
# import ArcFace
# import numpy as np
# import keras
# from keras import layers, Sequential
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import pandas as pd
# import tensorflow as tf
# from keras.models import load_model


# st.title('Face Recognition System')
# os.makedirs('data', exist_ok=True)
# name_list = os.listdir('data')

# # Data Collection
# # st.sidebar.title('Data Collection')
# webcam_channel = st.sidebar.selectbox(
#     'Webcam Channel:',
#     ('Select Channel', '0', '1', '2', '3')
# )
# name_person = st.text_input('Name of the Person:')
# img_number = st.number_input('Number of Images:', 50)
# FRAME_WINDOW = st.image([])

# if not webcam_channel == 'Select Channel':
#     take_img = st.button('Take Images')
#     if take_img:
#         if len(name_list) != 0:
#             for i in name_list:
#                 if i == name_person:
#                     st.warning('The Name is Already Exist!!')
#                     break
#         os.mkdir(f'data/{name_person}')
#         st.success(f'{name_person} added Successfully')

#         if len(os.listdir(f'data/{name_person}')) == 0:
#             face_classifier = cv2.CascadeClassifier(
#                 'haarcascade_frontalface_default.xml')
#             cap = cv2.VideoCapture(int(webcam_channel))
#             count = 0
#             while True:
#                 success, img = cap.read()
#                 if not success:
#                     st.error('[INFO] Cam NOT working!!')
#                     break

#                 # Save Image
#                 cv2.imwrite(f'data/{name_person}/{count}.jpg', img)
#                 st.success(f'[INFO] Successfully Saved {count}.jpg')
#                 count += 1

#                 faces = face_classifier.detectMultiScale(img)
#                 for (x, y, w, h) in faces:
#                     cv2.rectangle(
#                         img, (x, y), (x+w, y+h),
#                         (0, 255, 0), 2
#                     )

#                 FRAME_WINDOW.image(img, channels='BGR')
#                 if count == img_number:
#                     st.success(f'[INFO] Collected {img_number} Images')
#                     break

#             FRAME_WINDOW.image([])
#             cap.release()
#             cv2.destroyAllWindows()

# else:
#     st.warning('[INFO] Select Camera Channel')


# # 2nd Stage - Normalize Image Data
# st.sidebar.title('Normalize Image Data')
# if st.sidebar.button('Normalize'):
#     path_to_dir = "data"
#     path_to_save = 'norm_data'

#     Flage = True
#     detector = MTCNN()

#     class_list_update = []
#     if os.path.exists(path_to_save):
#         class_list_save = os.listdir(path_to_save)
#         class_list_dir = os.listdir(path_to_dir)
#         class_list_update = list(set(class_list_dir) ^ set(class_list_save))
#     else:
#         os.makedirs(path_to_save)

#     if len(class_list_update) == 0:
#         if (set(class_list_dir) == set(class_list_save)):
#             Flage = False
#         else:
#             class_list = os.listdir(path_to_dir)
#     else:
#         class_list = class_list_update

#     if Flage:
#         class_list = sorted(class_list)
#         for name in class_list:
#             st.success(f"[INFO] Class '{name}' Started Normalising")
#             img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')

#             # Create Save Folder
#             save_folder = os.path.join(path_to_save, name)
#             os.makedirs(save_folder, exist_ok=True)

#             for img_path in img_list:
#                 img = cv2.imread(img_path)

#                 detections = detector.detect_faces(img)

#                 if len(detections) > 0:
#                     right_eye = detections[0]['keypoints']['right_eye']
#                     left_eye = detections[0]['keypoints']['left_eye']
#                     bbox = detections[0]['box']
#                     norm_img_roi = alignment_procedure(
#                         img, left_eye, right_eye, bbox)

#                     # Save Norm ROI
#                     cv2.imwrite(
#                         f'{save_folder}/{os.path.split(img_path)[1]}', norm_img_roi)
#                     # st.success(f'[INFO] Successfully Normalised {img_path}')

#                 else:
#                     st.warning(f'[INFO] Not detected Eyes in {img_path}')

#             st.success(
#                 f"[INFO] All Normalised Images from '{name}' Saved in '{path_to_save}'")
#         st.success(
#             f'[INFO] Successfully Normalised All Images from {len(os.listdir(path_to_dir))} Classes\n')

#     else:
#         st.warning('[INFO] Already Normalized All Data..')


# # 3rd Stage - Train Model
# st.sidebar.title('Train Model')
# if st.sidebar.button('Train Model'):
#     path_to_dir = "norm_data"
#     path_to_save = 'model.h5'

#     # Load ArcFace Model
#     model = ArcFace.loadModel()
#     target_size = model.layers[0].input_shape[0][1:3]

#     # Variable for store img Embedding
#     x = []
#     y = []

#     names = os.listdir(path_to_dir)
#     names = sorted(names)
#     class_number = len(names)

#     for name in names:
#         img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')
#         img_list = sorted(img_list)
#         st.success(f'[INFO] Started Embedding {name} Class')

#         for img_path in img_list:
#             img = cv2.imread(img_path)
#             img_resize = cv2.resize(img, target_size)
#             # what this line doing? must?
#             img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
#             img_pixels = np.expand_dims(img_pixels, axis=0)
#             img_norm = img_pixels/255  # normalize input in [0, 1]
#             img_embedding = model.predict(img_norm)[0]

#             x.append(img_embedding)
#             y.append(name)

#         st.success(f'[INFO] Completed Embedding {name} Class')
#     st.success('[INFO] All Image Data Embedding Completed...')

#     # Model Training
#     # DataFrame
#     df = pd.DataFrame(x, columns=np.arange(512))
#     st.dataframe(df)
#     df['names'] = y

#     x = df.copy()
#     y = x.pop('names')
#     st.write(y)
#     y, _ = y.factorize()
#     x = x.astype('float64')
#     y = keras.utils.np_utils.to_categorical(y)

#     # Train Deep Neural Network
#     x_train, x_test, y_train, y_test = train_test_split(x, y,
#                                                         test_size=0.2,
#                                                         random_state=0)

#     model = Sequential([
#         layers.Dense(1024, activation='relu', input_shape=[512]),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(class_number, activation="softmax")
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     # Add a checkpoint callback to store the checkpoint that has the highest
#     # validation accuracy.
#     checkpoint_path = path_to_save
#     checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  monitor='val_accuracy',
#                                                  verbose=1,
#                                                  save_best_only=True,
#                                                  mode='max')
#     earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
#                                                   patience=20)

#     st.success('[INFO] Model Training Started ...')
#     # Start training
#     history = model.fit(x_train, y_train,
#                         epochs=200,
#                         batch_size=16,
#                         validation_data=(x_test, y_test),
#                         callbacks=[checkpoint, earlystopping])

#     st.success('[INFO] Model Training Completed')
#     st.success(f'[INFO] Model Successfully Saved in ./{path_to_save}')

#     # Plot History
#     metric_loss = history.history['loss']
#     metric_val_loss = history.history['val_loss']
#     metric_accuracy = history.history['accuracy']
#     metric_val_accuracy = history.history['val_accuracy']

#     # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
#     epochs = range(len(metric_loss))

#     # Plot the Graph.
#     plt.plot(epochs, metric_loss, 'blue', label=metric_loss)
#     plt.plot(epochs, metric_val_loss, 'red', label=metric_val_loss)
#     plt.plot(epochs, metric_accuracy, 'blue', label=metric_accuracy)
#     plt.plot(epochs, metric_val_accuracy, 'green', label=metric_val_accuracy)

#     # Add title to the plot.
#     plt.title(str('Model Metrics'))

#     # Add legend to the plot.
#     plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])

#     # If the plot already exist, remove
#     plot_png = os.path.exists('metrics.png')
#     if plot_png:
#         os.remove('metrics.png')
#         plt.savefig('metrics.png', bbox_inches='tight')
#     else:
#         plt.savefig('metrics.png', bbox_inches='tight')
#     st.success('[INFO] Successfully Saved metrics.png')


# # 4th Stage - Inference
# st.sidebar.title('Inference')
# # Confidence
# threshold = st.sidebar.slider('Model Confidence:', 0.01, 0.99, 0.6)

# if st.sidebar.button('Run/Stop'):
#     class_names = os.listdir('data')
#     class_names = sorted(class_names)

#     if not webcam_channel == 'Select Channel':
#         path_saved_model = "model.h5"
#         cap = cv2.VideoCapture(int(webcam_channel))
#         # Load MTCNN
#         detector = MTCNN()
#         arcface_model = ArcFace.loadModel()
#         target_size = arcface_model.layers[0].input_shape[0][1:3]
#         # Load saved FaceRecognition Model
#         face_rec_model = load_model(path_saved_model, compile=True)

#         while True:
#             success, img = cap.read()
#             if not success:
#                 st.warning('[INFO] Error with Camera')
#                 break

#             detections = detector.detect_faces(img)
#             if len(detections) > 0:
#                 for detect in detections:
#                     right_eye = detect['keypoints']['right_eye']
#                     left_eye = detect['keypoints']['left_eye']
#                     bbox = detect['box']
#                     xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), \
#                         int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])
#                     norm_img_roi = alignment_procedure(
#                         img, left_eye, right_eye, bbox)

#                     img_resize = cv2.resize(norm_img_roi, target_size)
#                     # what this line doing? must?
#                     img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
#                     img_pixels = np.expand_dims(img_pixels, axis=0)
#                     img_norm = img_pixels/255  # normalize input in [0, 1]
#                     img_embedding = arcface_model.predict(img_norm)[0]

#                     data = pd.DataFrame(
#                         [img_embedding], columns=np.arange(512))

#                     predict = face_rec_model.predict(data)[0]
#                     # print(predict)
#                     if max(predict) > threshold:
#                         pose_class = class_names[predict.argmax()]
#                     else:
#                         pose_class = 'Unkown Person'

#                     # Show Result
#                     cv2.rectangle(
#                         img, (xmin, ymin), (xmax, ymax),
#                         (0, 255, 0), 2
#                     )
#                     cv2.putText(
#                         img, f'{pose_class}',
#                         (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN,
#                         2, (255, 0, 255), 2
#                     )
#                     st.text(pose_class)

#             else:
#                 st.warning('[INFO] Eyes Not Detected!!')

#             FRAME_WINDOW.image(img, channels='BGR')

#         FRAME_WINDOW.image([])
#         st.success('[INFO] Inference on Videostream is Ended...')

#     else:
#         st.warning('[INFO] Select Camera Channel')


import streamlit as st
import cv2
import os
from datetime import datetime
import pandas as pd
from my_utils import alignment_procedure
from mtcnn import MTCNN
import glob
import ArcFace
import numpy as np
import keras
from keras import layers, Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import hashlib
# ... other required imports

# 2. Authentication Functions
def initialize_users_db():
    """Initialize or load users DataFrame"""
    if os.path.exists('users.csv'):
        return pd.read_csv('users.csv')
    else:
        return pd.DataFrame(columns=['username', 'password', 'role', 'email'])

def hash_password(password):
    """Hash password for security"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_credentials(username, password):
    """Verify user credentials"""
    users_df = initialize_users_db()
    hashed_pw = hash_password(password)
    user_row = users_df[users_df['username'] == username]
    if not user_row.empty and user_row.iloc[0]['password'] == hashed_pw:
        return True, user_row.iloc[0]['role']
    return False, None

def signup_user(username, password, role, email):
    """Register new user"""
    users_df = initialize_users_db()
    if username in users_df['username'].values:
        return False, "Username already exists"
    
    new_user = {
        'username': username,
        'password': hash_password(password),
        'role': role,
        'email': email
    }
    users_df = pd.concat([users_df, pd.DataFrame([new_user])], ignore_index=True)
    users_df.to_csv('users.csv', index=False)
    return True, "User registered successfully"
# Initialize attendance DataFrame
def initialize_attendance_df():
    if os.path.exists('attendance.csv'):
        return pd.read_csv('attendance.csv')
    else:
        return pd.DataFrame(columns=['Date', 'Subject', 'Enrollment_Number', 'Student_Name', 'Time', 'Status'])

# Save attendance record
def save_attendance(subject, enrollment_number, student_name):
    df = initialize_attendance_df()
    current_time = datetime.now()
    
    # Check if attendance already marked for this student, subject and date
    today = current_time.strftime('%Y-%m-%d')
    existing_record = df[
        (df['Date'] == today) & 
        (df['Subject'] == subject) & 
        (df['Enrollment_Number'] == enrollment_number)
    ]
    
    if existing_record.empty:
        new_record = {
            'Date': today,
            'Subject': subject,
            'Enrollment_Number': enrollment_number,
            'Student_Name': student_name,
            'Time': current_time.strftime('%H:%M:%S'),
            'Status': 'Present'
        }
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        df.to_csv('attendance.csv', index=False)
        return True
    return False

# Get absentees
def get_absentees(subject, date):
    if not os.path.exists('students_master.csv'):
        st.error("Students master list not found!")
        return None
    
    master_df = pd.read_csv('students_master.csv')
    if not os.path.exists('attendance.csv'):
        return master_df  # All students are absent
        
    attendance_df = pd.read_csv('attendance.csv')
    
    # Filter attendance for given subject and date
    present_students = attendance_df[
        (attendance_df['Subject'] == subject) & 
        (attendance_df['Date'] == date)
    ]['Enrollment_Number'].unique()
    
    # Get absentees
    absentees = master_df[~master_df['Enrollment_Number'].isin(present_students)]
    return absentees
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

def send_absence_email(student_email, student_name, subject_name, date):
    """Send absence notification email to student"""
    try:
        # Email configuration
        sender_email = st.secrets["email"]["sender_email"]
        sender_password = st.secrets["email"]["sender_password"]
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = student_email
        msg['Subject'] = f'Absence Notification - {subject_name}'
        
        # Email body
        body = f"""Dear {student_name},

This is to inform you that you were marked absent for {subject_name} on {date}.

Please ensure regular attendance in classes.

Best regards,
College Administration"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to SMTP server (Gmail example)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        
        # Send email
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email to {student_name}: {str(e)}")
        return False
# 4. Authentication UI
def show_login_page():
    """Display login form"""
    st.title("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login", key="login_button"):
        is_valid, role = check_credentials(username, password)
        if is_valid:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['role'] = role
            st.success(f"Logged in successfully as {role}")
            st.rerun()
        else:
            st.error("Invalid username or password")

def show_signup_page():
    """Display signup form"""
    st.title("Sign Up")
    username = st.text_input("Username", key="signup_username")
    password = st.text_input("Password", type="password", key="signup_password")
    email = st.text_input("Email", key="signup_email")
    role = st.selectbox("Role", ["student", "teacher"], key="signup_role")
    
    if st.button("Sign Up", key="signup_button"):
        success, message = signup_user(username, password, role, email)
        if success:
            st.success(message)
        else:
            st.error(message)
# Main Application
def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    # Show authentication pages if not logged in
    if not st.session_state['logged_in']:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            show_login_page()
        with tab2:
            show_signup_page()
        return
    st.title('Face Recognition Attendance System')
    os.makedirs('data', exist_ok=True)
    name_list = os.listdir('data')

    # Sidebar Navigation
    st.sidebar.title('Navigation')
    # Sidebar Navigation based on role
    if st.session_state['role'] == 'teacher':
        app_mode = st.sidebar.selectbox('Choose Mode',
            ['Data Collection', 'Normalize Data', 'Train Model', 'Take Attendance', 'View Reports'])
    else:  # student role
        app_mode = 'Take Attendance'
        st.sidebar.text("Mode: Take Attendance")
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['role'] = None
        st.rerun()
    # app_mode = st.sidebar.selectbox('Choose Mode',
    #     ['Data Collection', 'Normalize Data', 'Train Model', 'Take Attendance', 'View Reports'])

    # Common Settings
    webcam_channel = st.sidebar.selectbox(
        'Webcam Channel:',
        ('Select Channel', '0', '1', '2', '3')
    )

    if app_mode == 'Data Collection':
        st.header('Student Registration')
        
        enrollment_number = st.text_input('Enrollment Number:')
        name_person = st.text_input('Student Name:')
        subject_name = st.text_input('Subject Name:')
        img_number = st.number_input('Number of Images:', 50)
        FRAME_WINDOW = st.image([])

        if not webcam_channel == 'Select Channel':
            take_img = st.button('Register Student')
            if take_img:
                if not enrollment_number or not name_person:
                    st.warning('Please fill enrollment number and student name!')
                else:
                    folder_name = f"{enrollment_number}_{name_person}"
                    os.makedirs(f'data/{folder_name}', exist_ok=True)
                    
                    # Save to master list
                    if not os.path.exists('students_master.csv'):
                        master_df = pd.DataFrame(columns=['Enrollment_Number', 'Student_Name'])
                    else:
                        master_df = pd.read_csv('students_master.csv')
                    
                    if not master_df[master_df['Enrollment_Number'] == enrollment_number].empty:
                        st.warning('This enrollment number already exists!')
                    else:
                        new_student = {
                            'Enrollment_Number': enrollment_number,
                            'Student_Name': name_person
                        }
                        master_df = pd.concat([master_df, pd.DataFrame([new_student])], ignore_index=True)
                        master_df.to_csv('students_master.csv', index=False)
                        
                        # Capture Images
                        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                        cap = cv2.VideoCapture(int(webcam_channel))
                        count = 0
                        
                        while True:
                            success, img = cap.read()
                            if not success:
                                st.error('[INFO] Camera not working!')
                                break

                            faces = face_classifier.detectMultiScale(img)
                            if len(faces) > 0:
                                # Save Image only when face is detected
                                cv2.imwrite(f'data/{folder_name}/{count}.jpg', img)
                                st.success(f'[INFO] Saved image {count + 1}/{img_number}')
                                count += 1

                                for (x, y, w, h) in faces:
                                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                            FRAME_WINDOW.image(img, channels='BGR')
                            if count >= img_number:
                                st.success(f'Registration completed for {name_person}')
                                break

                        FRAME_WINDOW.image([])
                        cap.release()
                        cv2.destroyAllWindows()

    elif app_mode == 'Normalize Data':
        st.header('Normalize Image Data')
        if st.button('Start Normalization'):
            path_to_dir = "data"
            path_to_save = 'norm_data'
            os.makedirs(path_to_save, exist_ok=True)
            
            detector = MTCNN()
            class_list = os.listdir(path_to_dir)
            
            for name in class_list:
                st.info(f"Normalizing images for {name}")
                img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')
                save_folder = os.path.join(path_to_save, name)
                os.makedirs(save_folder, exist_ok=True)

                for img_path in img_list:
                    img = cv2.imread(img_path)
                    detections = detector.detect_faces(img)

                    if len(detections) > 0:
                        right_eye = detections[0]['keypoints']['right_eye']
                        left_eye = detections[0]['keypoints']['left_eye']
                        bbox = detections[0]['box']
                        norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)
                        cv2.imwrite(f'{save_folder}/{os.path.split(img_path)[1]}', norm_img_roi)
                    else:
                        st.warning(f'No face detected in {img_path}')
                
            st.success('Normalization completed!')

    elif app_mode == 'Train Model':
        st.header('Train Recognition Model')
        if st.button('Start Training'):
            # Load ArcFace Model
            model = ArcFace.loadModel()
            target_size = model.layers[0].input_shape[0][1:3]

            # Prepare data
            x = []
            y = []
            names = sorted(os.listdir('norm_data'))
            
            for name in names:
                st.info(f'Processing {name}')
                img_list = glob.glob(os.path.join('norm_data', name) + '/*')
                
                for img_path in img_list:
                    img = cv2.imread(img_path)
                    img_resize = cv2.resize(img, target_size)
                    img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
                    img_pixels = np.expand_dims(img_pixels, axis=0)
                    img_norm = img_pixels/255
                    img_embedding = model.predict(img_norm)[0]
                    
                    x.append(img_embedding)
                    y.append(name)

            # Prepare training data
            df = pd.DataFrame(x, columns=np.arange(512))
            df['names'] = y
            
            x = df.drop('names', axis=1)
            y = pd.factorize(df['names'])[0]
            y = keras.utils.np_utils.to_categorical(y)

            # Split data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            # Build model
            model = Sequential([
                layers.Dense(1024, activation='relu', input_shape=[512]),
                layers.Dense(512, activation='relu'),
                layers.Dense(len(names), activation="softmax")
            ])

            model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

            # Train model
            history = model.fit(x_train, y_train,
                              epochs=100,
                              batch_size=32,
                              validation_data=(x_test, y_test),
                              callbacks=[
                                  keras.callbacks.ModelCheckpoint(
                                      'model.h5',
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      mode='max'),
                                  keras.callbacks.EarlyStopping(
                                      monitor='val_accuracy',
                                      patience=20)
                              ])

            st.success('Model training completed!')
    elif app_mode == 'Take Attendance':
        st.header('Take Attendance')
        
        subject_name = st.text_input('Subject Name:')
        confidence_threshold = st.slider('Confidence Threshold:', 0.0, 1.0, 0.6)
        FRAME_WINDOW = st.image([])

        if st.button('Start Attendance'):
            if not subject_name:
                st.warning('Please enter subject name!')
            elif not os.path.exists('model.h5'):
                st.error('Model not found! Please train the model first.')
            else:
                try:
                    # Load all required models
                    detector = MTCNN()
                    arcface_model = ArcFace.loadModel()
                    face_rec_model = load_model('model.h5')
                    
                    # Get list of registered students and their folders
                    registered_students = os.listdir('data')
                    if not registered_students:
                        st.error('No registered students found!')
                        return
                        
                    # Sort to ensure consistent ordering with model training
                    registered_students = sorted(registered_students)
                    
                    # Create mapping of class index to student info
                    class_to_student = {}
                    for idx, student_folder in enumerate(registered_students):
                        try:
                            # Verify folder name format
                            parts = student_folder.split('_', 1)  # Split on first underscore only
                            if len(parts) != 2:
                                st.warning(f"Skipping invalid folder name: {student_folder}")
                                continue
                            enrollment, name = parts
                            class_to_student[idx] = {
                                'enrollment': enrollment,
                                'name': name
                            }
                            st.text(class_to_student)
                            st.text(idx)
                            st.text(class_to_student[idx])
                        except Exception as e:
                            st.warning(f"Error processing folder {student_folder}: {str(e)}")
                            continue
                    
                    cap = cv2.VideoCapture(int(webcam_channel))
                    recognized_students = set()
                    
                    # Status indicators
                    status_placeholder = st.empty()
                    recognition_placeholder = st.empty()
                    
                    while True:
                        success, img = cap.read()
                        if not success:
                            st.error('Camera not working!')
                            break

                        status_placeholder.text('Scanning for faces...')
                        detections = detector.detect_faces(img)
                        
                        if len(detections) > 0:
                            for detection in detections:
                                try:
                                    bbox = detection['box']
                                    right_eye = detection['keypoints']['right_eye']
                                    left_eye = detection['keypoints']['left_eye']
                                    
                                    # Process detected face
                                    norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)
                                    img_resize = cv2.resize(norm_img_roi, arcface_model.layers[0].input_shape[0][1:3])
                                    img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
                                    img_pixels = np.expand_dims(img_pixels, axis=0)
                                    img_norm = img_pixels/255
                                    
                                    # Get face embedding and prediction
                                    img_embedding = arcface_model.predict(img_norm)[0]
                                    prediction = face_rec_model.predict(np.array([img_embedding]))[0]
                                    
                                    # Show prediction confidence
                                    max_confidence = max(prediction)
                                    recognition_placeholder.text(f'Recognition confidence: {max_confidence:.2%}')
                                    
                                    if max_confidence > confidence_threshold:
                                        class_idx = np.argmax(prediction)
                                        
                                        # Get student info from mapping
                                        if class_idx in class_to_student:
                                            student = class_to_student[class_idx]
                                            enrollment = student['enrollment']
                                            name = student['name']
                                            st.text(name)
                                            
                                            # Mark attendance if not already marked
                                            if enrollment not in recognized_students:
                                                if save_attendance(subject_name, enrollment, name):
                                                    recognized_students.add(enrollment)
                                                    st.success(f"✅ Marked attendance for {name} ({enrollment})")
                                            
                                            # Draw rectangle and name
                                            cv2.rectangle(img, 
                                                       (int(bbox[0]), int(bbox[1])), 
                                                       (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
                                                       (0, 255, 0), 2)
                                            
                                            # Display name and confidence
                                            label = f"{name} ({max_confidence:.1%})"
                                            cv2.putText(img, label,
                                                      (int(bbox[0]), int(bbox[1] - 10)), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                                      (0, 255, 0), 2)
                                        else:
                                            st.warning(f"Unmatched class index: {class_idx}")
                                    else:
                                        # Draw red rectangle for unrecognized face
                                        cv2.rectangle(img, 
                                                   (int(bbox[0]), int(bbox[1])), 
                                                   (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
                                                   (0, 0, 255), 2)
                                        cv2.putText(img, "Unknown",
                                                  (int(bbox[0]), int(bbox[1] - 10)), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                                  (0, 0, 255), 2)
                                
                                except Exception as e:
                                    st.error(f"Error processing detection: {str(e)}")
                                    continue
                        
                        # Display the image
                        FRAME_WINDOW.image(img, channels='BGR')
                        
                        # Show current attendance count
                        status_placeholder.text(f'Students marked present: {len(recognized_students)}')

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    # Clean up
                    if 'cap' in locals():
                        cap.release()
                    cv2.destroyAllWindows()
                    status_placeholder.empty()
                    recognition_placeholder.empty()
    # elif app_mode == 'Take Attendance':
    #     st.header('Take Attendance')
        
    #     subject_name = st.text_input('Subject Name:')
    #     confidence_threshold = st.slider('Confidence Threshold:', 0.0, 1.0, 0.6)
    #     FRAME_WINDOW = st.image([])

    #     if st.button('Start Attendance'):
    #         if not subject_name:
    #             st.warning('Please enter subject name!')
    #         elif not os.path.exists('model.h5'):
    #             st.error('Model not found! Please train the model first.')
    #         else:
    #             detector = MTCNN()
    #             arcface_model = ArcFace.loadModel()
    #             face_rec_model = load_model('model.h5')
                
    #             cap = cv2.VideoCapture(int(webcam_channel))
    #             recognized_students = set()

    #             while True:
    #                 success, img = cap.read()
    #                 if not success:
    #                     st.error('Camera not working!')
    #                     break

    #                 detections = detector.detect_faces(img)
    #                 if len(detections) > 0:
    #                     for detection in detections:
    #                         bbox = detection['box']
    #                         right_eye = detection['keypoints']['right_eye']
    #                         left_eye = detection['keypoints']['left_eye']
                            
    #                         norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)
    #                         img_resize = cv2.resize(norm_img_roi, arcface_model.layers[0].input_shape[0][1:3])
    #                         img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
    #                         img_pixels = np.expand_dims(img_pixels, axis=0)
    #                         img_norm = img_pixels/255
                            
    #                         img_embedding = arcface_model.predict(img_norm)[0]
    #                         prediction = face_rec_model.predict(np.array([img_embedding]))[0]
    #                         st.text(prediction)
    #                         if max(prediction) > confidence_threshold:
    #                             class_idx = np.argmax(prediction)
    #                             st.text(class_idx)
    #                             student_info = os.listdir('data')[class_idx].split('_')
    #                             st.text(student_info)
    #                             enrollment = student_info[0]
    #                             st.text(enrollment)
    #                             name = student_info[1]
    #                             st.text(name)
                                
    #                             # if enrollment not in recognized_students:
    #                             if save_attendance(subject_name, enrollment, name):
    #                                 recognized_students.add(enrollment)
    #                                 st.success(f"Marked attendance for {name}")
                                
    #                             # Draw rectangle and name
    #                             cv2.rectangle(img, 
    #                                        (int(bbox[0]), int(bbox[1])), 
    #                                        (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
    #                                        (0, 255, 0), 2)
    #                             cv2.putText(img, name, 
    #                                       (int(bbox[0]), int(bbox[1] - 10)), 
    #                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
    #                                       (0, 255, 0), 2)

    #                 FRAME_WINDOW.image(img, channels='BGR')

    #             cap.release()
    #             cv2.destroyAllWindows()
    elif app_mode == 'View Reports':
        st.header('Attendance Reports')
        
        # Date Selection
        report_date = st.date_input('Select Date')
        
        # Get unique subjects from attendance records
        if os.path.exists('attendance.csv'):
            attendance_df = pd.read_csv('attendance.csv')
            subjects = sorted(attendance_df['Subject'].unique())
            
            if len(subjects) > 0:
                selected_subject = st.selectbox('Select Subject', subjects)
                
                if st.button('Generate Report'):
                    try:
                        # Show present students
                        present_students = attendance_df[
                            (attendance_df['Date'] == report_date.strftime('%Y-%m-%d')) & 
                            (attendance_df['Subject'] == selected_subject)
                        ]
                        
                        # Get total students from master list with email addresses
                        master_df = pd.read_csv('students_master.csv')
                        total_students = len(master_df)
                        
                        # Display Statistics in columns
                        col1, col2, col3 = st.columns(3)
                        present_count = len(present_students)
                        absent_count = total_students - present_count
                        attendance_percentage = (present_count / total_students) * 100 if total_students > 0 else 0
                        
                        col1.metric("Total Students", total_students)
                        col2.metric("Present", present_count)
                        col3.metric("Absent", absent_count)
                        
                        # Display attendance percentage with color coding
                        if attendance_percentage >= 75:
                            st.success(f"Attendance Percentage: {attendance_percentage:.2f}%")
                        elif attendance_percentage >= 60:
                            st.warning(f"Attendance Percentage: {attendance_percentage:.2f}%")
                        else:
                            st.error(f"Attendance Percentage: {attendance_percentage:.2f}%")
                        
                        # Present Students Details
                        if not present_students.empty:
                            st.subheader('Present Students')
                            # Sort by time
                            present_students = present_students.sort_values('Time')
                            st.dataframe(
                                present_students[['Enrollment_Number', 'Student_Name', 'Time']]
                                .style.set_properties(**{'background-color': '#90EE90'})
                            )
                            
                            # Export option for present students
                            csv_present = present_students.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Present Students List",
                                csv_present,
                                f"present_students_{selected_subject}_{report_date}.csv",
                                "text/csv",
                                key='download-present-csv'
                            )
                        
                        # Absent Students Details and Email Notifications
                        absentees = get_absentees(selected_subject, report_date.strftime('%Y-%m-%d'))
                        if absentees is not None and not absentees.empty:
                            st.subheader('Absent Students')
                            
                            # Create DataFrame with email column
                            absent_df = pd.merge(
                                absentees,
                                master_df[['Enrollment_Number', 'Email']],
                                on='Enrollment_Number',
                                how='left'
                            )
                            
                            # Display absentee list with email addresses
                            st.dataframe(
                                absent_df[['Enrollment_Number', 'Student_Name', 'Email']]
                                .style.set_properties(**{'background-color': '#FFB6C1'})
                            )
                            
                            # Email notification section
                            st.subheader('Email Notifications')
                            email_col1, email_col2 = st.columns(2)
                            
                            with email_col1:
                                sender_email = st.text_input("Sender Email", type="default")
                            with email_col2:
                                sender_password = st.text_input("Email Password", type="password")
                            
                            email_subject = st.text_input("Email Subject", 
                                f"Absence Notification - {selected_subject} ({report_date.strftime('%Y-%m-%d')})")
                            
                            email_body = st.text_area("Email Body", f"""Dear {{student_name}},
                                This is to inform you that you were marked absent for {selected_subject} on {report_date.strftime('%Y-%m-%d')}.

                                Please ensure regular attendance in classes.

                                Best regards,
                                College Administration""")
                            
                            if st.button('Send Absence Notifications'):
                                if not sender_email or not sender_password:
                                    st.error("Please provide email credentials")
                                else:
                                    with st.spinner('Sending email notifications...'):
                                        success_count = 0
                                        fail_count = 0
                                        
                                        # Progress bar
                                        progress_bar = st.progress(0)
                                        total_emails = len(absent_df)
                                        
                                        for idx, student in absent_df.iterrows():
                                            try:
                                                if pd.isna(student['vivekraina33.vr@gmail.com']):
                                                    st.warning(f"No email address found for {student['Student_Name']}")
                                                    fail_count += 1
                                                    continue
                                                
                                                # Create message
                                                msg = MIMEMultipart()
                                                msg['From'] = sender_email
                                                msg['To'] = student['vivekraina33.vr@gmail.com']
                                                msg['Subject'] = email_subject
                                                
                                                # Personalize email body
                                                personalized_body = email_body.replace(
                                                    "{student_name}", student['Student_Name'])
                                                msg.attach(MIMEText(personalized_body, 'plain'))
                                                
                                                # Connect to SMTP server (Gmail example)
                                                server = smtplib.SMTP('smtp.gmail.com', 587)
                                                server.starttls()
                                                server.login(sender_email, sender_password)
                                                
                                                # Send email
                                                server.send_message(msg)
                                                server.quit()
                                                
                                                success_count += 1
                                                st.success(f"Email sent to {student['Student_Name']}")
                                                
                                            except Exception as e:
                                                fail_count += 1
                                                st.error(f"Failed to send email to {student['Student_Name']}: {str(e)}")
                                            
                                            # Update progress bar
                                            progress_bar.progress((idx + 1) / total_emails)
                                        
                                        # Final status
                                        if success_count > 0:
                                            st.success(f"Successfully sent {success_count} email notifications")
                                        if fail_count > 0:
                                            st.error(f"Failed to send {fail_count} email notifications")
                            
                            # Export options for absent students
                            csv_absent = absent_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Absent Students List",
                                csv_absent,
                                f"absent_students_{selected_subject}_{report_date}.csv",
                                "text/csv",
                                key='download-absent-csv'
                            )
                        
                        # Visualization Section
                        st.subheader('Attendance Visualization')
                        
                        # Pie Chart for Present/Absent
                        fig_pie, ax_pie = plt.subplots()
                        ax_pie.pie(
                            [present_count, absent_count],
                            labels=['Present', 'Absent'],
                            autopct='%1.1f%%',
                            colors=['#90EE90', '#FFB6C1']
                        )
                        ax_pie.set_title(f'Attendance Distribution for {selected_subject}')
                        st.pyplot(fig_pie)
                        
                        # Time Analysis
                        if not present_students.empty:
                            st.subheader('Attendance Timing Analysis')
                            
                            # Convert Time to datetime
                            present_students['Time'] = pd.to_datetime(present_students['Time'])
                            present_students['Hour'] = present_students['Time'].dt.hour
                            
                            # Create hour-wise distribution
                            hour_dist = present_students.groupby('Hour').size().reset_index(name='Count')
                            
                            # Bar chart for timing distribution
                            fig_time, ax_time = plt.subplots(figsize=(10, 6))
                            ax_time.bar(hour_dist['Hour'], hour_dist['Count'])
                            ax_time.set_xlabel('Hour of Day')
                            ax_time.set_ylabel('Number of Students')
                            ax_time.set_title('Attendance Timing Distribution')
                            plt.xticks(hour_dist['Hour'])
                            st.pyplot(fig_time)
                        
                        # Monthly Trend Analysis
                        st.subheader('Monthly Attendance Trend')
                        # Get attendance for the current month
                        current_month = pd.to_datetime(report_date).strftime('%Y-%m')
                        month_attendance = attendance_df[
                            (attendance_df['Subject'] == selected_subject) &
                            (pd.to_datetime(attendance_df['Date']).dt.strftime('%Y-%m') == current_month)
                        ]
                        
                        if not month_attendance.empty:
                            # Group by date and count attendance
                            daily_counts = month_attendance.groupby('Date').size().reset_index(name='Count')
                            daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
                            
                            # Line plot for monthly trend
                            fig_trend, ax_trend = plt.subplots(figsize=(12, 6))
                            ax_trend.plot(daily_counts['Date'], daily_counts['Count'], marker='o')
                            ax_trend.set_xlabel('Date')
                            ax_trend.set_ylabel('Number of Students')
                            ax_trend.set_title('Daily Attendance Trend')
                            plt.xticks(rotation=45)
                            st.pyplot(fig_trend)
                            
                        # Export Complete Report
                        st.subheader('Export Complete Report')
                        
                        # Prepare report data
                        report_data = {
                            'Report Date': report_date.strftime('%Y-%m-%d'),
                            'Subject': selected_subject,
                            'Total Students': total_students,
                            'Present': present_count,
                            'Absent': absent_count,
                            'Attendance Percentage': f"{attendance_percentage:.2f}%"
                        }
                        
                        report_df = pd.DataFrame([report_data])
                        
                        # Export to Excel with multiple sheets
                        def create_excel_report():
                            with pd.ExcelWriter('attendance_report.xlsx', engine='xlsxwriter') as writer:
                                # Summary Sheet
                                report_df.to_excel(writer, sheet_name='Summary', index=False)
                                
                                # Present Students Sheet
                                if not present_students.empty:
                                    present_students.to_excel(writer, sheet_name='Present Students', index=False)
                                
                                # Absent Students Sheet
                                if absentees is not None and not absentees.empty:
                                    absent_df.to_excel(writer, sheet_name='Absent Students', index=False)
                            
                            with open('attendance_report.xlsx', 'rb') as f:
                                return f.read()
                        
                        excel_report = create_excel_report()
                        st.download_button(
                            "Download Complete Excel Report",
                            excel_report,
                            f"attendance_report_{selected_subject}_{report_date}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key='download-excel'
                        )
                        
                    except Exception as e:
                        st.error(f"An error occurred while generating the report: {str(e)}")
                        
            else:
                st.warning("No attendance records found. Please take attendance first.")
        else:
            st.error("No attendance records found. Please take attendance first.")
    # elif app_mode == 'View Reports':
    #     st.header('Attendance Reports')
        
    #     # Date Selection
    #     report_date = st.date_input('Select Date')
        
    #     # Get unique subjects from attendance records
    #     if os.path.exists('attendance.csv'):
    #         attendance_df = pd.read_csv('attendance.csv')
    #         subjects = sorted(attendance_df['Subject'].unique())
            
    #         if len(subjects) > 0:
    #             selected_subject = st.selectbox('Select Subject', subjects)
                
    #             if st.button('Generate Report'):
    #                 try:
    #                     # Show present students
    #                     present_students = attendance_df[
    #                         (attendance_df['Date'] == report_date.strftime('%Y-%m-%d')) & 
    #                         (attendance_df['Subject'] == selected_subject)
    #                     ]
                        
    #                     # Get total students from master list
    #                     master_df = pd.read_csv('students_master.csv')
    #                     total_students = len(master_df)
                        
    #                     # Display Statistics in columns
    #                     col1, col2, col3 = st.columns(3)
    #                     present_count = len(present_students)
    #                     absent_count = total_students - present_count
    #                     attendance_percentage = (present_count / total_students) * 100 if total_students > 0 else 0
                        
    #                     col1.metric("Total Students", total_students)
    #                     col2.metric("Present", present_count)
    #                     col3.metric("Absent", absent_count)
                        
    #                     # Display attendance percentage with color coding
    #                     if attendance_percentage >= 75:
    #                         st.success(f"Attendance Percentage: {attendance_percentage:.2f}%")
    #                     elif attendance_percentage >= 60:
    #                         st.warning(f"Attendance Percentage: {attendance_percentage:.2f}%")
    #                     else:
    #                         st.error(f"Attendance Percentage: {attendance_percentage:.2f}%")
                        
    #                     # Present Students Details
    #                     if not present_students.empty:
    #                         st.subheader('Present Students')
    #                         # Sort by time
    #                         present_students = present_students.sort_values('Time')
    #                         st.dataframe(
    #                             present_students[['Enrollment_Number', 'Student_Name', 'Time']]
    #                             .style.set_properties(**{'background-color': '#90EE90'})
    #                         )
                            
    #                         # Export option for present students
    #                         csv_present = present_students.to_csv(index=False).encode('utf-8')
    #                         st.download_button(
    #                             "Download Present Students List",
    #                             csv_present,
    #                             f"present_students_{selected_subject}_{report_date}.csv",
    #                             "text/csv",
    #                             key='download-present-csv'
    #                         )
                        
    #                     # Absent Students Details
    #                     absentees = get_absentees(selected_subject, report_date.strftime('%Y-%m-%d'))
    #                     if absentees is not None and not absentees.empty:
    #                         st.subheader('Absent Students')
    #                         st.dataframe(
    #                             absentees[['Enrollment_Number', 'Student_Name']]
    #                             .style.set_properties(**{'background-color': '#FFB6C1'})
    #                         )
                            
    #                         # Export option for absent students
    #                         csv_absent = absentees.to_csv(index=False).encode('utf-8')
    #                         st.download_button(
    #                             "Download Absent Students List",
    #                             csv_absent,
    #                             f"absent_students_{selected_subject}_{report_date}.csv",
    #                             "text/csv",
    #                             key='download-absent-csv'
    #                         )
                        
    #                     # Visualization Section
    #                     st.subheader('Attendance Visualization')
                        
    #                     # Pie Chart for Present/Absent
    #                     fig_pie, ax_pie = plt.subplots()
    #                     ax_pie.pie(
    #                         [present_count, absent_count],
    #                         labels=['Present', 'Absent'],
    #                         autopct='%1.1f%%',
    #                         colors=['#90EE90', '#FFB6C1']
    #                     )
    #                     ax_pie.set_title(f'Attendance Distribution for {selected_subject}')
    #                     st.pyplot(fig_pie)
                        
    #                     # Time Analysis
    #                     if not present_students.empty:
    #                         st.subheader('Attendance Timing Analysis')
                            
    #                         # Convert Time to datetime
    #                         present_students['Time'] = pd.to_datetime(present_students['Time'])
    #                         present_students['Hour'] = present_students['Time'].dt.hour
                            
    #                         # Create hour-wise distribution
    #                         hour_dist = present_students.groupby('Hour').size().reset_index(name='Count')
                            
    #                         # Bar chart for timing distribution
    #                         fig_time, ax_time = plt.subplots(figsize=(10, 6))
    #                         ax_time.bar(hour_dist['Hour'], hour_dist['Count'])
    #                         ax_time.set_xlabel('Hour of Day')
    #                         ax_time.set_ylabel('Number of Students')
    #                         ax_time.set_title('Attendance Timing Distribution')
    #                         plt.xticks(hour_dist['Hour'])
    #                         st.pyplot(fig_time)
                        
    #                     # Monthly Trend Analysis
    #                     st.subheader('Monthly Attendance Trend')
    #                     # Get attendance for the current month
    #                     current_month = pd.to_datetime(report_date).strftime('%Y-%m')
    #                     month_attendance = attendance_df[
    #                         (attendance_df['Subject'] == selected_subject) &
    #                         (pd.to_datetime(attendance_df['Date']).dt.strftime('%Y-%m') == current_month)
    #                     ]
                        
    #                     if not month_attendance.empty:
    #                         # Group by date and count attendance
    #                         daily_counts = month_attendance.groupby('Date').size().reset_index(name='Count')
    #                         daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
                            
    #                         # Line plot for monthly trend
    #                         fig_trend, ax_trend = plt.subplots(figsize=(12, 6))
    #                         ax_trend.plot(daily_counts['Date'], daily_counts['Count'], marker='o')
    #                         ax_trend.set_xlabel('Date')
    #                         ax_trend.set_ylabel('Number of Students')
    #                         ax_trend.set_title('Daily Attendance Trend')
    #                         plt.xticks(rotation=45)
    #                         st.pyplot(fig_trend)
                            
    #                     # Export Complete Report
    #                     st.subheader('Export Complete Report')
                        
    #                     # Prepare report data
    #                     report_data = {
    #                         'Report Date': report_date.strftime('%Y-%m-%d'),
    #                         'Subject': selected_subject,
    #                         'Total Students': total_students,
    #                         'Present': present_count,
    #                         'Absent': absent_count,
    #                         'Attendance Percentage': f"{attendance_percentage:.2f}%"
    #                     }
                        
    #                     report_df = pd.DataFrame([report_data])
                        
    #                     # Export to Excel with multiple sheets
    #                     def create_excel_report():
    #                         with pd.ExcelWriter('attendance_report.xlsx', engine='xlsxwriter') as writer:
    #                             # Summary Sheet
    #                             report_df.to_excel(writer, sheet_name='Summary', index=False)
                                
    #                             # Present Students Sheet
    #                             if not present_students.empty:
    #                                 present_students.to_excel(writer, sheet_name='Present Students', index=False)
                                
    #                             # Absent Students Sheet
    #                             if absentees is not None and not absentees.empty:
    #                                 absentees.to_excel(writer, sheet_name='Absent Students', index=False)
                            
    #                         with open('attendance_report.xlsx', 'rb') as f:
    #                             return f.read()
                        
    #                     excel_report = create_excel_report()
    #                     st.download_button(
    #                         "Download Complete Excel Report",
    #                         excel_report,
    #                         f"attendance_report_{selected_subject}_{report_date}.xlsx",
    #                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    #                         key='download-excel'
    #                     )
                        
    #                 except Exception as e:
    #                     st.error(f"An error occurred while generating the report: {str(e)}")
                        
    #         else:
    #             st.warning("No attendance records found. Please take attendance first.")
    #     else:
    #         st.error("No attendance records found. Please take attendance first.")
main()
    # elif app_mode == 'View Reports':
    #     st.header('Attendance Reports')
        
    #     report_date = st.date_input('Select Date')
        
    #     # Get unique subjects from attendance records
    #     if os.path.exists('attendance.csv'):
    #         attendance_df = pd.read_csv('attendance.csv')
    #         subjects = attendance_df['Subject'].unique()
    #         selected_subject = st.selectbox('Select Subject', subjects)
            
    #         if st.button('Generate Report'):
    #             # Show present students
    #             present_students = attendance_df[
    #                 (attendance_df['Date'] == report_date.strftime('%Y-%m-%d')) & 
    #                 (attendance_df['Subject'] == selected_subject)
    #             ]
                
    #             if not present_students.empty:
    #                 st.subheader('Present Students')
    #                 st.dataframe(present_students[['Enrollment_Number', 'Student_Name', 'Time']])
                
    #             # Show absent students
    #             absentees = get_absentees(selected_subject, report_date.strftime('%Y-%m-%d'))
    #             if absentees is not None and not absentees.empty:
    #                 st.subheader('Absent Students')
    #                 st.dataframe(absentees[['Enrollment_Number', 'Student_Name']])
                
    #             # Calculate statistics
    #             total_students = len(pd.read_csv('students_master.csv'))
    #             present_count = len(present_students)
    #             absent_count = total_students - present_count
                
    #             col1, col2, col3 = st.columns(3)
    #             col1.metric("Total Students", total_students)
    #             col2.metric("Present", present_count)
    #             col3.metric("Absent", absent_count)
                
    #             # Attendance percentage
    #             attendance_percentage = (present_count / total_students) * 100 if total