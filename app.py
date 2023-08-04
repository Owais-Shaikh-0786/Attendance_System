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

# Import mode images into a list
folderModePath = 'Resources'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
print(len(imgModeList))

# Load the encoded face data
print("Loading Encode File ...")
with open('encodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "attendance-system-0786.appspot.com"
})
db = firestore.client()

# Function to get attendance status based on time
def get_attendance_status(class_start_time, attendance_cutoff_time):
    now = datetime.now().time()

    if class_start_time < now <= attendance_cutoff_time:
        return "Present"
    elif now >= attendance_cutoff_time:
        return "Late"

# Function to get class start time and attendance cutoff time from user
def get_class_time_input():
    class_start_time_str = input("Enter the class start time (format: HH:MM): ")
    attendance_cutoff_time_str = input("Enter the attendance cutoff time (format: HH:MM): ")

    class_start_time = datetime.strptime(class_start_time_str, "%H:%M").time()
    attendance_cutoff_time = datetime.strptime(attendance_cutoff_time_str, "%H:%M").time()

    return class_start_time, attendance_cutoff_time

# Function to retrieve student details from Firestore
def retrieve_student_details():
    student_details = []

    students_collection = db.collection("studentsDetails")

    for doc_ref in students_collection.stream():
        student_data = doc_ref.to_dict()

        if student_data:
            student_details.append(student_data)

    return student_details

# Asynchronous function to mark attendance for a student
async def markattendanceasync(student_details, stu_id, class_start_time, attendance_cutoff_time):

    student_data = None
    for student in student_details:
        if student.get("id") == stu_id:
            student_data = student
            break

    if student_data is None:
        print(f"Student with ID {stu_id} not found.")
        return

    student_name = student_data.get("name")
    now = datetime.now()

    year_folder = now.strftime('%Y')
    month_folder = now.strftime('%B')
    date_folder = now.strftime('%d')

    student_status = get_attendance_status(class_start_time, attendance_cutoff_time)

    attendance_ref = db.collection("Attendance")
    year_doc_ref = attendance_ref.document(year_folder)
    month_doc_ref = year_doc_ref.collection(month_folder)
    student_doc_ref = month_doc_ref.document(date_folder)

    current_time = now.strftime("%H:%M:%S")

    student_folder_ref = student_doc_ref.collection("Students").document(stu_id)

    student_data = {
        'name': student_name,
        'time': current_time,
        'status': student_status,
    }
    student_folder_ref.set(student_data)

# Main asynchronous function
async def main():
    student_details = retrieve_student_details()
    class_start_time, attendance_cutoff_time = get_class_time_input()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    imgBackground = cv2.imread('background.png')

    while True:
        success, img = cap.read()

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        imgBackground[162:162 + 480, 55:55 + 640] = img
        imgBackground[44:44 + 633, 800:800 + 414] = imgModeList[0]

        if faceCurFrame:
            tasks = []
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                matchIndex = np.argmin(faceDis)

                if min(faceDis) >= 0.4:
                    print("Student name: unknown")
                    continue

                if matches[matchIndex]:
                    stu_id = studentIds[matchIndex]
                    print("Student name:", studentIds[matchIndex])
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    tasks.append(markattendanceasync(student_details, stu_id, class_start_time, attendance_cutoff_time))
                    imgBackground[44:44 + 633, 800:800 + 414] = imgModeList[1]

            if tasks:
                await asyncio.gather(*tasks)

        cv2.imshow("Attendance System", imgBackground)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
