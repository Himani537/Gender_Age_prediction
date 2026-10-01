"""
Gender and Age Prediction using OpenCV DNN
"""

import cv2 as cv
import time
import argparse
import os


# ---------------------------------------------------------
# FACE DETECTION
# ---------------------------------------------------------
def getFaceBox(net, frame, conf_threshold=0.4):

    frameOpencvDnn = frame.copy()

    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    blob = cv.dnn.blobFromImage(
        frameOpencvDnn,
        1.0,
        (300, 300),
        [104, 117, 123],
        True,
        False
    )

    net.setInput(blob)
    detections = net.forward()

    bboxes = []

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:

            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frameWidth - 1, x2)
            y2 = min(frameHeight - 1, y2)

            bboxes.append([x1, y1, x2, y2])

            cv.rectangle(
                frameOpencvDnn,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    return frameOpencvDnn, bboxes


# ---------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Gender and Age Recognition using OpenCV"
)

parser.add_argument(
    "-i",
    help="Path to input image or video file. Skip this argument to use webcam."
)

args = parser.parse_args()


# ---------------------------------------------------------
# MODEL FILES
# ---------------------------------------------------------
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"


# ---------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------
MODEL_MEAN_VALUES = (
    78.4263377603,
    87.7689143744,
    114.895847746
)

ageList = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)"
]

genderList = [
    "Male",
    "Female"
]


# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
print("Loading models...")

ageNet = cv.dnn.readNetFromCaffe(
    ageProto,
    ageModel
)

genderNet = cv.dnn.readNetFromCaffe(
    genderProto,
    genderModel
)

faceNet = cv.dnn.readNet(
    faceModel,
    faceProto
)

print("Models loaded successfully!")


# ---------------------------------------------------------
# OPEN CAMERA / IMAGE / VIDEO
# ---------------------------------------------------------
if args.i:

    # Image or video file
    cap = cv.VideoCapture(args.i)

else:

    # Webcam
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    # Set camera resolution
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)


if not cap.isOpened():

    print("ERROR: Cannot open camera or input file.")
    exit()


padding = 20


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
while True:

    t = time.time()

    hasFrame, frame = cap.read()

    if not hasFrame:

        print("Cannot read frame.")
        break


    # Detect face
    frameFace, bboxes = getFaceBox(
        faceNet,
        frame,
        conf_threshold=0.4
    )


    # -----------------------------------------------------
    # NO FACE
    # -----------------------------------------------------
    if not bboxes:

        print("No face detected, checking next frame...")

        cv.imshow("Age Gender Demo", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        continue


    # -----------------------------------------------------
    # FACE FOUND
    # -----------------------------------------------------
    for bbox in bboxes:

        x1, y1, x2, y2 = bbox

        face = frame[
            max(0, y1 - padding):
            min(y2 + padding, frame.shape[0] - 1),

            max(0, x1 - padding):
            min(x2 + padding, frame.shape[1] - 1)
        ]


        # Check face image
        if face.size == 0:
            continue


        # -------------------------------------------------
        # PREPARE FACE
        # -------------------------------------------------
        blob = cv.dnn.blobFromImage(
            face,
            1.0,
            (227, 227),
            MODEL_MEAN_VALUES,
            swapRB=False
        )


        # -------------------------------------------------
        # GENDER PREDICTION
        # -------------------------------------------------
        genderNet.setInput(blob)

        genderPreds = genderNet.forward()

        gender = genderList[
            genderPreds[0].argmax()
        ]

        genderConfidence = genderPreds[0].max()

        print(
            "Gender: {}, confidence = {:.3f}".format(
                gender,
                genderConfidence
            )
        )


        # -------------------------------------------------
        # AGE PREDICTION
        # -------------------------------------------------
        ageNet.setInput(blob)

        agePreds = ageNet.forward()

        age = ageList[
            agePreds[0].argmax()
        ]

        ageConfidence = agePreds[0].max()

        print(
            "Age: {}, confidence = {:.3f}".format(
                age,
                ageConfidence
            )
        )


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------
        label = "{}, {}".format(
            gender,
            age
        )

        cv.putText(
            frameFace,
            label,
            (x1, y1 - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv.LINE_AA
        )


    # -----------------------------------------------------
    # SHOW WINDOW
    # -----------------------------------------------------
    cv.imshow(
        "Age Gender Demo",
        frameFace
    )


    print(
        "Time: {:.3f}".format(
            time.time() - t
        )
    )


    # Press Q to quit
    if cv.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------------------------------
# RELEASE
# ---------------------------------------------------------
cap.release()

cv.destroyAllWindows()









# '''
# PyPower Projects
# Detect Gender and Age using Artificial Intelligence

# '''

# #Usage 
# # Step 1 : Go to command prompt and set working directory where the gender_age.py file is stored
# # Step 2 : Execute the following command to detect from image: python gender_age.py -i 1.jpg  
# # Step 3 : Execute the following command to detect from webcam: python gender_age.py


# # Import required modules
# import cv2 as cv
# import math
# import time
# import argparse

# def getFaceBox(net, frame, conf_threshold=0.7):
#     frameOpencvDnn = frame.copy()
#     frameHeight = frameOpencvDnn.shape[0]
#     frameWidth = frameOpencvDnn.shape[1]
#     blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

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
#             bboxes.append([x1, y1, x2, y2])
#             cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
#     return frameOpencvDnn, bboxes


# parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
# parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')

# args = parser.parse_args()

# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"

# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"

# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']

# # Load network
# ageNet = cv.dnn.readNetFromCaffe(ageProto,ageModel)
# genderNet = cv.dnn.readNetFromCaffe(genderProto,genderModel)
# faceNet = cv.dnn.readNet(faceModel,faceProto)

# # Open a video file or an image file or a camera stream
# cap = cv.VideoCapture(args.i if args.i else 0)
# padding = 20
# while cv.waitKey(1) < 0:
#     # Read frame
#     t = time.time()
#     hasFrame, frame = cap.read()
#     if not hasFrame:
#         cv.waitKey()
#         break
#     frameFace, bboxes = getFaceBox(faceNet, frame)
#     if not bboxes:
#         print("No face Detected, Checking next frame")
#         continue

#     for bbox in bboxes:
#         # print(bbox)
#         face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

#         blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
#         genderNet.setInput(blob)
#         genderPreds = genderNet.forward()
#         gender = genderList[genderPreds[0].argmax()]
        
#         print("Gender : {}, confidence = {:.3f}".format(gender, genderPreds[0].max()))

#         ageNet.setInput(blob)
#         agePreds = ageNet.forward()
#         age = ageList[agePreds[0].argmax()]
        
#         print("Age : {}, confidence = {:.3f}".format(age, agePreds[0].max()))

#         label = "{},{}".format(gender, age)
#         cv.putText(frameFace, label, (bbox[0]-5, bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255), 2, cv.LINE_AA)
#         cv.imshow("Age Gender Demo", frameFace)
#         name = args.i
#         cv.imwrite('./detected/'+name,frameFace)
#     print("Time : {:.3f}".format(time.time() - t))

