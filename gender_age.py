# """
# Gender and Age Prediction using OpenCV DNN
# """

# import cv2 as cv
# import time
# import argparse
# import os


# # ---------------------------------------------------------
# # FACE DETECTION
# # ---------------------------------------------------------
# def getFaceBox(net, frame, conf_threshold=0.4):

#     frameOpencvDnn = frame.copy()

#     frameHeight = frameOpencvDnn.shape[0]
#     frameWidth = frameOpencvDnn.shape[1]

#     blob = cv.dnn.blobFromImage(
#         frameOpencvDnn,
#         1.0,
#         (300, 300),
#         [104, 117, 123],
#         True,
#         False
#     )

#     net.setInput(blob)
#     detections = net.forward()

#     bboxes = []

#     for i in range(detections.shape[2]):

#         confidence = detections[0, 0, i, 2]

#         if confidence > conf_threshold:

#             x1 = int(detections[0, 0, i, 3] * frameWidth)
#             y1 = int(detections[0, 0, i, 4] * frameHeight)
#             x2 = int(detections[0, 0, i, 5] * frameWidth)
#             y2 = int(detections[0, 0, i, 6] * frameHeight)

#             # Keep coordinates inside image
#             x1 = max(0, x1)
#             y1 = max(0, y1)
#             x2 = min(frameWidth - 1, x2)
#             y2 = min(frameHeight - 1, y2)

#             bboxes.append([x1, y1, x2, y2])

#             cv.rectangle(
#                 frameOpencvDnn,
#                 (x1, y1),
#                 (x2, y2),
#                 (0, 255, 0),
#                 2
#             )

#     return frameOpencvDnn, bboxes


# # ---------------------------------------------------------
# # ARGUMENTS
# # ---------------------------------------------------------
# parser = argparse.ArgumentParser(
#     description="Gender and Age Recognition using OpenCV"
# )

# parser.add_argument(
#     "-i",
#     help="Path to input image or video file. Skip this argument to use webcam."
# )

# args = parser.parse_args()


# # ---------------------------------------------------------
# # MODEL FILES
# # ---------------------------------------------------------
# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"

# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"

# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"


# # ---------------------------------------------------------
# # MODEL SETTINGS
# # ---------------------------------------------------------
# MODEL_MEAN_VALUES = (
#     78.4263377603,
#     87.7689143744,
#     114.895847746
# )

# ageList = [
#     "(0-2)",
#     "(4-6)",
#     "(8-12)",
#     "(15-20)",
#     "(25-32)",
#     "(38-43)",
#     "(48-53)",
#     "(60-100)"
# ]

# genderList = [
#     "Male",
#     "Female"
# ]


# # ---------------------------------------------------------
# # LOAD MODELS
# # ---------------------------------------------------------
# print("Loading models...")

# ageNet = cv.dnn.readNetFromCaffe(
#     ageProto,
#     ageModel
# )

# genderNet = cv.dnn.readNetFromCaffe(
#     genderProto,
#     genderModel
# )

# faceNet = cv.dnn.readNet(
#     faceModel,
#     faceProto
# )

# print("Models loaded successfully!")


# # ---------------------------------------------------------
# # OPEN CAMERA / IMAGE / VIDEO
# # ---------------------------------------------------------
# if args.i:

#     # Image or video file
#     cap = cv.VideoCapture(args.i)

# else:

#     # Webcam
#     cap = cv.VideoCapture(0, cv.CAP_DSHOW)

#     # Set camera resolution
#     cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)


# if not cap.isOpened():

#     print("ERROR: Cannot open camera or input file.")
#     exit()


# padding = 20


# # ---------------------------------------------------------
# # MAIN LOOP
# # ---------------------------------------------------------
# while True:

#     t = time.time()

#     hasFrame, frame = cap.read()

#     if not hasFrame:

#         print("Cannot read frame.")
#         break


#     # Detect face
#     frameFace, bboxes = getFaceBox(
#         faceNet,
#         frame,
#         conf_threshold=0.4
#     )


#     # -----------------------------------------------------
#     # NO FACE
#     # -----------------------------------------------------
#     if not bboxes:

#         print("No face detected, checking next frame...")

#         cv.imshow("Age Gender Demo", frame)

#         if cv.waitKey(1) & 0xFF == ord("q"):
#             break

#         continue


#     # -----------------------------------------------------
#     # FACE FOUND
#     # -----------------------------------------------------
#     for bbox in bboxes:

#         x1, y1, x2, y2 = bbox

#         face = frame[
#             max(0, y1 - padding):
#             min(y2 + padding, frame.shape[0] - 1),

#             max(0, x1 - padding):
#             min(x2 + padding, frame.shape[1] - 1)
#         ]


#         # Check face image
#         if face.size == 0:
#             continue


#         # -------------------------------------------------
#         # PREPARE FACE
#         # -------------------------------------------------
#         blob = cv.dnn.blobFromImage(
#             face,
#             1.0,
#             (227, 227),
#             MODEL_MEAN_VALUES,
#             swapRB=False
#         )


#         # -------------------------------------------------
#         # GENDER PREDICTION
#         # -------------------------------------------------
#         genderNet.setInput(blob)

#         genderPreds = genderNet.forward()

#         gender = genderList[
#             genderPreds[0].argmax()
#         ]

#         genderConfidence = genderPreds[0].max()

#         print(
#             "Gender: {}, confidence = {:.3f}".format(
#                 gender,
#                 genderConfidence
#             )
#         )


#         # -------------------------------------------------
#         # AGE PREDICTION
#         # -------------------------------------------------
#         ageNet.setInput(blob)

#         agePreds = ageNet.forward()

#         age = ageList[
#             agePreds[0].argmax()
#         ]

#         ageConfidence = agePreds[0].max()

#         print(
#             "Age: {}, confidence = {:.3f}".format(
#                 age,
#                 ageConfidence
#             )
#         )


#         # -------------------------------------------------
#         # DISPLAY RESULT
#         # -------------------------------------------------
#         label = "{}, {}".format(
#             gender,
#             age
#         )

#         cv.putText(
#             frameFace,
#             label,
#             (x1, y1 - 10),
#             cv.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 0, 255),
#             2,
#             cv.LINE_AA
#         )


#     # -----------------------------------------------------
#     # SHOW WINDOW
#     # -----------------------------------------------------
#     cv.imshow(
#         "Age Gender Demo",
#         frameFace
#     )


#     print(
#         "Time: {:.3f}".format(
#             time.time() - t
#         )
#     )


#     # Press Q to quit
#     if cv.waitKey(1) & 0xFF == ord("q"):
#         break


# # ---------------------------------------------------------
# # RELEASE
# # ---------------------------------------------------------
# cap.release()

# cv.destroyAllWindows()









# # '''
# # PyPower Projects
# # Detect Gender and Age using Artificial Intelligence

# # '''

# # #Usage 
# # # Step 1 : Go to command prompt and set working directory where the gender_age.py file is stored
# # # Step 2 : Execute the following command to detect from image: python gender_age.py -i 1.jpg  
# # # Step 3 : Execute the following command to detect from webcam: python gender_age.py


# # # Import required modules
# # import cv2 as cv
# # import math
# # import time
# # import argparse

# # def getFaceBox(net, frame, conf_threshold=0.7):
# #     frameOpencvDnn = frame.copy()
# #     frameHeight = frameOpencvDnn.shape[0]
# #     frameWidth = frameOpencvDnn.shape[1]
# #     blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

# #     net.setInput(blob)
# #     detections = net.forward()
# #     bboxes = []
# #     for i in range(detections.shape[2]):
# #         confidence = detections[0, 0, i, 2]
# #         if confidence > conf_threshold:
# #             x1 = int(detections[0, 0, i, 3] * frameWidth)
# #             y1 = int(detections[0, 0, i, 4] * frameHeight)
# #             x2 = int(detections[0, 0, i, 5] * frameWidth)
# #             y2 = int(detections[0, 0, i, 6] * frameHeight)
# #             bboxes.append([x1, y1, x2, y2])
# #             cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
# #     return frameOpencvDnn, bboxes


# # parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
# # parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')

# # args = parser.parse_args()

# # faceProto = "opencv_face_detector.pbtxt"
# # faceModel = "opencv_face_detector_uint8.pb"

# # ageProto = "age_deploy.prototxt"
# # ageModel = "age_net.caffemodel"

# # genderProto = "gender_deploy.prototxt"
# # genderModel = "gender_net.caffemodel"

# # MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# # ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# # genderList = ['Male', 'Female']

# # # Load network
# # ageNet = cv.dnn.readNetFromCaffe(ageProto,ageModel)
# # genderNet = cv.dnn.readNetFromCaffe(genderProto,genderModel)
# # faceNet = cv.dnn.readNet(faceModel,faceProto)

# # # Open a video file or an image file or a camera stream
# # cap = cv.VideoCapture(args.i if args.i else 0)
# # padding = 20
# # while cv.waitKey(1) < 0:
# #     # Read frame
# #     t = time.time()
# #     hasFrame, frame = cap.read()
# #     if not hasFrame:
# #         cv.waitKey()
# #         break
# #     frameFace, bboxes = getFaceBox(faceNet, frame)
# #     if not bboxes:
# #         print("No face Detected, Checking next frame")
# #         continue

# #     for bbox in bboxes:
# #         # print(bbox)
# #         face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

# #         blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
# #         genderNet.setInput(blob)
# #         genderPreds = genderNet.forward()
# #         gender = genderList[genderPreds[0].argmax()]
        
# #         print("Gender : {}, confidence = {:.3f}".format(gender, genderPreds[0].max()))

# #         ageNet.setInput(blob)
# #         agePreds = ageNet.forward()
# #         age = ageList[agePreds[0].argmax()]
        
# #         print("Age : {}, confidence = {:.3f}".format(age, agePreds[0].max()))

# #         label = "{},{}".format(gender, age)
# #         cv.putText(frameFace, label, (bbox[0]-5, bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255), 2, cv.LINE_AA)
# #         cv.imshow("Age Gender Demo", frameFace)
# #         name = args.i
# #         cv.imwrite('./detected/'+name,frameFace)
# #     print("Time : {:.3f}".format(time.time() - t))

"""
Gender and Age Prediction - Streamlit UI
Run with:  streamlit run app.py
Place this file in the same folder as the model files.
"""

import os
import tempfile
import time

import cv2 as cv
import numpy as np
import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gender & Age Detection",
    page_icon="🧑",
    layout="wide",
)

# ---------------------------------------------------------
# MODEL FILES (relative to this script's folder)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_PROTO = os.path.join(BASE_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(BASE_DIR, "opencv_face_detector_uint8.pb")
AGE_PROTO = os.path.join(BASE_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(BASE_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(BASE_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(BASE_DIR, "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

AGE_LIST = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)",
]
GENDER_LIST = ["Male", "Female"]

# BGR colors for drawing
COLOR_MALE = (255, 140, 0)
COLOR_FEMALE = (180, 60, 255)


# ---------------------------------------------------------
# LOAD MODELS (cached so they load only once)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    required = [
        FACE_PROTO, FACE_MODEL, AGE_PROTO,
        AGE_MODEL, GENDER_PROTO, GENDER_MODEL,
    ]
    missing = [os.path.basename(p) for p in required if not os.path.exists(p)]
    if missing:
        return None, missing

    age_net = cv.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    gender_net = cv.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
    face_net = cv.dnn.readNet(FACE_MODEL, FACE_PROTO)
    return (face_net, age_net, gender_net), []


# ---------------------------------------------------------
# FACE DETECTION
# ---------------------------------------------------------
def get_face_boxes(net, frame, conf_threshold=0.5):
    h, w = frame.shape[:2]

    blob = cv.dnn.blobFromImage(
        frame, 1.0, (300, 300), [104, 117, 123], True, False
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2, float(confidence)])
    return boxes


# ---------------------------------------------------------
# DRAW LABEL
# ---------------------------------------------------------
def draw_label(img, text, x1, y1, color):
    scale = max(0.5, img.shape[1] / 1400)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), base = cv.getTextSize(
        text, cv.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    top = max(0, y1 - th - base - 8)
    cv.rectangle(img, (x1, top), (x1 + tw + 10, top + th + base + 8), color, -1)
    cv.putText(
        img, text, (x1 + 5, top + th + 3),
        cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
        thickness, cv.LINE_AA,
    )


# ---------------------------------------------------------
# ANALYZE ONE FRAME
# ---------------------------------------------------------
def analyze_frame(frame, models, face_conf=0.5, padding=20):
    face_net, age_net, gender_net = models
    output = frame.copy()
    results = []

    boxes = get_face_boxes(face_net, frame, face_conf)
    h, w = frame.shape[:2]
    line_w = max(2, int(round(h / 200)))

    for idx, (x1, y1, x2, y2, det_conf) in enumerate(boxes, start=1):
        face = frame[
            max(0, y1 - padding): min(y2 + padding, h),
            max(0, x1 - padding): min(x2 + padding, w),
        ]
        if face.size == 0:
            continue

        blob = cv.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
        )

        gender_net.setInput(blob)
        g_preds = gender_net.forward()[0]
        gender = GENDER_LIST[g_preds.argmax()]
        g_conf = float(g_preds.max())

        age_net.setInput(blob)
        a_preds = age_net.forward()[0]
        age = AGE_LIST[a_preds.argmax()]
        a_conf = float(a_preds.max())

        color = COLOR_MALE if gender == "Male" else COLOR_FEMALE
        cv.rectangle(output, (x1, y1), (x2, y2), color, line_w)
        draw_label(output, f"{gender}, {age}", x1, y1, color)

        results.append({
            "Face": idx,
            "Gender": gender,
            "Gender Confidence": f"{g_conf * 100:.1f}%",
            "Age Range": age,
            "Age Confidence": f"{a_conf * 100:.1f}%",
            "Detection Confidence": f"{det_conf * 100:.1f}%",
        })

    return output, results


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def to_rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def decode_image(uploaded_file):
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv.imdecode(data, cv.IMREAD_COLOR)


def show_results(annotated, results, key_prefix="img"):
    col_img, col_info = st.columns([3, 2])

    with col_img:
        st.image(to_rgb(annotated), caption="Result", use_container_width=True)

    with col_info:
        if not results:
            st.warning("No face detected. Try lowering the detection threshold.")
        else:
            st.success(f"{len(results)} face(s) detected")
            st.dataframe(results, use_container_width=True, hide_index=True)

        ok, buf = cv.imencode(".png", annotated)
        if ok:
            st.download_button(
                "⬇️ Download result image",
                data=buf.tobytes(),
                file_name="detected.png",
                mime="image/png",
                key=f"{key_prefix}_download",
            )

        if st.button("💾 Save to 'detected' folder", key=f"{key_prefix}_save"):
            out_dir = os.path.join(BASE_DIR, "detected")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"result_{int(time.time())}.png")
            cv.imwrite(path, annotated)
            st.info(f"Saved: {path}")


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🧑 Gender & Age Detection")
st.caption("Deep-learning face analysis using OpenCV DNN")

models, missing_files = load_models()
if models is None:
    st.error(
        "Missing model files in the app folder: " + ", ".join(missing_files)
    )
    st.stop()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Input source",
        ["📁 Upload Image", "📷 Camera Snapshot", "🎥 Live Webcam", "🎞️ Upload Video"],
    )

    face_conf = st.slider(
        "Face detection threshold", 0.1, 0.95, 0.5, 0.05,
        help="Lower = detects more faces (but more false positives).",
    )
    padding = st.slider(
        "Face padding (px)", 0, 60, 20, 5,
        help="Extra pixels around the face used for prediction.",
    )

    st.divider()
    st.caption(
        "Age is predicted as a range and is only an estimate. "
        "Accuracy depends on lighting, angle and image quality."
    )

# ---------------------------------------------------------
# MODE 1: UPLOAD IMAGE
# ---------------------------------------------------------
if mode == "📁 Upload Image":
    files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    for n, f in enumerate(files or []):
        img = decode_image(f)
        if img is None:
            st.error(f"Could not read {f.name}")
            continue

        st.subheader(f.name)
        with st.spinner("Analyzing..."):
            t0 = time.time()
            annotated, results = analyze_frame(img, models, face_conf, padding)
            elapsed = time.time() - t0

        show_results(annotated, results, key_prefix=f"upload_{n}")
        st.caption(f"Processed in {elapsed:.3f} s")
        st.divider()

# ---------------------------------------------------------
# MODE 2: CAMERA SNAPSHOT (works in browser)
# ---------------------------------------------------------
elif mode == "📷 Camera Snapshot":
    snap = st.camera_input("Take a picture")

    if snap is not None:
        img = decode_image(snap)
        with st.spinner("Analyzing..."):
            annotated, results = analyze_frame(img, models, face_conf, padding)
        show_results(annotated, results, key_prefix="snap")

# ---------------------------------------------------------
# MODE 3: LIVE WEBCAM
# ---------------------------------------------------------
elif mode == "🎥 Live Webcam":
    st.info(
        "Live webcam uses the camera of the computer running Streamlit. "
        "Untick the box to stop."
    )
    cam_index = st.number_input("Camera index", 0, 5, 0)
    run = st.checkbox("▶️ Start webcam")

    frame_slot = st.empty()
    info_slot = st.empty()

    if run:
        # CAP_DSHOW is Windows-only; fall back to default backend elsewhere
        cap = cv.VideoCapture(int(cam_index), cv.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv.VideoCapture(int(cam_index))

        if not cap.isOpened():
            st.error("Cannot open webcam.")
        else:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

            try:
                while run:
                    t0 = time.time()
                    ok, frame = cap.read()
                    if not ok:
                        st.warning("Cannot read frame from webcam.")
                        break

                    annotated, results = analyze_frame(
                        frame, models, face_conf, padding
                    )
                    frame_slot.image(to_rgb(annotated), use_container_width=True)

                    fps = 1.0 / max(time.time() - t0, 1e-6)
                    summary = ", ".join(
                        f"{r['Gender']} {r['Age Range']}" for r in results
                    ) or "No face detected"
                    info_slot.markdown(f"**FPS:** {fps:.1f} &nbsp;|&nbsp; **{summary}**")
            finally:
                cap.release()

# ---------------------------------------------------------
# MODE 4: UPLOAD VIDEO
# ---------------------------------------------------------
elif mode == "🎞️ Upload Video":
    vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    skip = st.slider("Process every Nth frame", 1, 10, 2)

    if vid is not None and st.button("▶️ Process video"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid.read())
        tfile.close()

        cap = cv.VideoCapture(tfile.name)
        total = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 1

        progress = st.progress(0.0)
        frame_slot = st.empty()
        info_slot = st.empty()

        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                i += 1
                if i % skip != 0:
                    continue

                annotated, results = analyze_frame(
                    frame, models, face_conf, padding
                )
                frame_slot.image(to_rgb(annotated), use_container_width=True)
                summary = ", ".join(
                    f"{r['Gender']} {r['Age Range']}" for r in results
                ) or "No face detected"
                info_slot.markdown(f"**Frame {i}/{total}** — {summary}")
                progress.progress(min(i / total, 1.0))
        finally:
            cap.release()
            try:
                os.remove(tfile.name)
            except OSError:
                pass

        progress.progress(1.0)
        st.success("Video processing finished.")
