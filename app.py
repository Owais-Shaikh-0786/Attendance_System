import cv2
import pickle
import face_recognition
import numpy as np
import cvzone
from datetime import datetime
import os
import asyncio
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Importing the mode images into a list
folderModePath = 'Resources'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
print(len(imgModeList))

print("Loading Encode File ...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "attendance-system-0786.appspot.com"
})
db = firestore.client()


def get_attendance_status(class_start_time, attendance_cutoff_time):
    now = datetime.now().time()

    if class_start_time < now <= attendance_cutoff_time:
        return "Present"
    elif now >= attendance_cutoff_time:
        return "Late"


def get_class_time_input():
    class_start_time_str = input("Enter the class start time (format: HH:MM): ")
    attendance_cutoff_time_str = input("Enter the attendance cutoff time (format: HH:MM): ")

    class_start_time = datetime.strptime(class_start_time_str, "%H:%M").time()
    attendance_cutoff_time = datetime.strptime(attendance_cutoff_time_str, "%H:%M").time()

    return class_start_time, attendance_cutoff_time


async def markattendanceasync(stu_id, class_start_time, attendance_cutoff_time):
    # Get the current date and time
    now = datetime.now()

    # Extract the year, month, and date from the current date
    year_folder = now.strftime('%Y')
    month_folder = now.strftime('%B')  # Use full month name (e.g., "July") instead of the numerical representation
    date_folder = now.strftime('%d')

    # Determine the student's attendance status based on the current time
    student_status = get_attendance_status(class_start_time, attendance_cutoff_time)

    # Create a reference to the "Attendance" collection in Firestore
    attendance_ref = db.collection("Attendance")

    # Create a reference to the main folder with the year as the document name
    year_doc_ref = attendance_ref.document(year_folder)

    # Create a reference to the subfolder with the month as the document name
    month_doc_ref = year_doc_ref.collection(month_folder)

    # Create a reference to the student's sub-subfolder with the student_id as the document name
    student_doc_ref = month_doc_ref.document(date_folder)

    # Get the current time
    current_time = now.strftime("%H:%M:%S")

    # Create a reference to the student's folder with the student name as the document name
    student_folder_ref = student_doc_ref.collection("Students").document(stu_id)

    # Update the student's attendance data for the current date and time
    student_data = {
        'name': stu_id,
        'status': student_status,
        'time': current_time
    }
    student_folder_ref.set(student_data)


async def main():
    # Ask the user for the class start time and attendance cutoff time
    class_start_time, attendance_cutoff_time = get_class_time_input()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    imgBackground = cv2.imread('background.png')

    while True:
        # Read frames from the webcam
        success, img = cap.read()

        # Resize the frame for faster face recognition
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings in the current frame
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        imgBackground[162:162 + 480, 55:55 + 640] = img

        imgBackground[44:44 + 633, 800:800 + 414] = imgModeList[0]

        if faceCurFrame:
            tasks = []
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                # Compare face encodings to known encodings
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                print("matches", matches)
                print("faceDis", faceDis)

                matchIndex = np.argmin(faceDis)

                print("matchIndex", matchIndex)
                print("face_location ", faceLoc)

                # If an unknown face is detected with the minimum similarity >= 0.4, print "unknown" and skip marking
                if min(faceDis) >= 0.4:
                    print("student name: unknown")
                    continue

                if matches[matchIndex]:
                    stu_id = studentIds[matchIndex]
                    print("student name:", studentIds[matchIndex])
                    # If a known face is detected
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    tasks.append(markattendanceasync(stu_id, class_start_time, attendance_cutoff_time))

                    imgBackground[44:44 + 633, 800:800 + 414] = imgModeList[1]

            if tasks:
                await asyncio.gather(*tasks)

        cv2.imshow("Attendance System", imgBackground)

        # If the 'q' key is pressed, break out of the loop and stop the video capture
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
