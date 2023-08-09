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


# Function to get class start time and attendance cutoff time from user
def get_class_time_input():
    # Prompt the user to input the class start time in the specified format
    class_start_time_str = input("Enter the class start time (format: HH:MM): ")

    # Prompt the user to input the attendance cutoff time in the specified format
    attendance_cutoff_time_str = input("Enter the attendance cutoff time (format: HH:MM): ")

    # Convert the input strings to time objects using the specified format
    class_start_time = datetime.strptime(class_start_time_str, "%H:%M").time()
    attendance_cutoff_time = datetime.strptime(attendance_cutoff_time_str, "%H:%M").time()

    # Return the class start time and attendance cutoff time as time objects
    return class_start_time, attendance_cutoff_time


# Function to get attendance status based on time
def get_attendance_status(class_start_time, attendance_cutoff_time):
    # Get the current time
    now = datetime.now().time()

    # Compare the current time with class start time and attendance cutoff time
    if class_start_time < now <= attendance_cutoff_time:
        # If the current time is after class start time and before or equal to attendance cutoff time,
        # return "Present" as the attendance status
        return "Present"
    elif now >= attendance_cutoff_time:
        # If the current time is equal to or after the attendance cutoff time,
        # return "Absent" as the attendance status
        return "Absent"


def retrieve_student_details():
    # Initialize an empty list to store student details
    student_details = []

    # Access the 'studentsDetails' collection from the database
    students_collection = db.collection("studentsDetails")

    # Iterate through each document reference in the collection
    for doc_ref in students_collection.stream():
        # Convert the document reference to a dictionary representing student data
        student_data = doc_ref.to_dict()

        # Check if the student data dictionary is not empty
        if student_data:
            # Append the non-empty student data dictionary to the list
            student_details.append(student_data)

    # Iterate through the list of student details dictionaries
    for student in student_details:
        # Retrieve the 'id' field from each student's data
        stu_id = student.get("id")

        # Print the student ID
        print("student_details id :", stu_id)

    # Return the list of student details dictionaries
    return student_details


def print_student_ids(folder_year, folder_month, folder_date):
    # Construct a reference to the specific student folder within the 'Attendance' collection
    student_folder_ref = (db.collection("Attendance").document(folder_year).collection(folder_month)
                          .document(folder_date).collection("Students"))

    # Iterate through student documents in the specified folder and print their IDs
    for student_doc in student_folder_ref.stream():
        print("Student ID:", student_doc.id)

    # Create a set of unique student IDs by iterating through the student documents
    student_ids_print = set(student_doc.id for student_doc in student_folder_ref.stream())

    # Return the set of unique student IDs
    return student_ids_print


def compare_student_ids():
    # Get the current date and time
    now = datetime.now()

    # Extract year, month, and date components from the current date
    year_folder = now.strftime('%Y')
    month_folder = now.strftime('%B')
    date_folder = now.strftime('%d')

    # Get a set of student IDs from the retrieved student details
    student_ids_details = set(student.get("id") for student in retrieve_student_details())

    # Get a set of student IDs from a specific attendance folder for the current date
    student_ids_print = print_student_ids(year_folder, month_folder, date_folder)

    # Find the set of missing student IDs by comparing the two sets
    missing_ids = student_ids_details - student_ids_print

    # Print the IDs of students who are missing from the attendance folder
    print("Missing IDs:", missing_ids)

    # Return the set of missing student IDs for further processing
    return missing_ids


# Function to retrieve student details from Firestore

async def markattendanceasync(student_details, stu_id, class_start_time, attendance_cutoff_time, status=None):
    # Find student data using the provided student ID
    student_data = None
    for student in student_details:
        if student.get("id") == stu_id:
            student_data = student
            break

    # If student data is not found, print an error message and return
    if student_data is None:
        print(f"Student with ID {stu_id} not found.")
        return

    # Extract relevant student information
    student_name = student_data.get("name")
    now = datetime.now()

    # Generate year, month, and date components from the current date
    year_folder = now.strftime('%Y')
    month_folder = now.strftime('%B')
    date_folder = now.strftime('%d')

    # Determine student's attendance status using provided status or default function
    student_status = status or get_attendance_status(class_start_time, attendance_cutoff_time)

    # Access Firestore references for attendance tracking
    attendance_ref = db.collection("Attendance")
    year_doc_ref = attendance_ref.document(year_folder)
    month_doc_ref = year_doc_ref.collection(month_folder)
    student_doc_ref = month_doc_ref.document(date_folder)

    # Get current time in HH:MM:SS format
    current_time = now.strftime("%H:%M:%S")

    # Create a reference to the student's attendance document
    student_folder_ref = student_doc_ref.collection("Students").document(stu_id)

    # Check if attendance status has already been set
    existing_data = student_folder_ref.get().to_dict()
    if existing_data and existing_data.get('status'):
        print(f"Attendance status already set for Student ID {stu_id}.")
        return

    # Prepare data to be stored in the student's attendance document
    student_data = {
        'name': student_name,
        'time': current_time,
        'status': student_status,
    }

    # Set the attendance data for the student in the database
    student_folder_ref.set(student_data)


# Main asynchronous function
async def main():
    # Retrieve student details and input class times
    student_details = retrieve_student_details()
    class_start_time, attendance_cutoff_time = get_class_time_input()

    # Initialize video capture and background image
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    imgBackground = cv2.imread('background.png')

    # Initialize variables for attendance checking
    attendance_checked = False

    while True:
        # Capture a frame from the webcam
        success, img = cap.read()

        # Resize and convert the captured frame
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect faces and encode current frame
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        # Update the background image with captured frame and mode information
        imgBackground[162:162 + 480, 55:55 + 640] = img
        imgBackground[44:44 + 633, 800:800 + 414] = imgModeList[0]

        if faceCurFrame:
            tasks = []
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                # Compare face encodings and find the best match
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                # Check if the face is recognized based on a threshold
                if min(faceDis) >= 0.4:
                    print("Student name: unknown")
                    continue

                if matches[matchIndex]:
                    stu_id = studentIds[matchIndex]
                    print("Student name:", studentIds[matchIndex])

                    # Update bounding box and mark attendance asynchronously
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    tasks.append(markattendanceasync(student_details, stu_id, class_start_time, attendance_cutoff_time))
                    imgBackground[44:44 + 633, 800:800 + 414] = imgModeList[1]

            if tasks:
                await asyncio.gather(*tasks)

        cv2.imshow("Attendance System", imgBackground)

        # Check and mark attendance once when the cutoff time is reached
        now = datetime.now()
        if not attendance_checked and now.time() > attendance_cutoff_time:
            compare_student_ids()
            missed_ids = compare_student_ids()

            # Mark absent students asynchronously
            for stu_id in missed_ids:
                await markattendanceasync(student_details, stu_id, class_start_time, attendance_cutoff_time,
                                          status="Absent")
            attendance_checked = True

        # Break the loop on pressing 'q' key
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
