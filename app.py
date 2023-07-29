import cv2
import pickle
import face_recognition
import numpy as np
import cvzone
from datetime import datetime
import os
import asyncio

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


async def markattendanceasync(stu_id):
    # Get the current date and time
    now = datetime.now()

    # Extract the year from the current date
    year_folder = now.strftime('%Y')

    # Create a directory path for the attendance records with the year as the folder name
    attendance_dir = os.path.join('../Attendance System/Attendance', year_folder)

    # Check if the attendance directory exists, and if not, create it
    if not os.path.exists(attendance_dir):
        os.makedirs(attendance_dir)

    # Create a path for the CSV file with the current date as the name
    csv_file = os.path.join(attendance_dir, f'{now.strftime("%Y-%m-%d")}.csv')

    # Check if the CSV file for the current date exists, and if not, create it and add the header
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write('Name,Time\n')

    # Read the existing lines from the CSV file to check if the name already exists in the attendance list
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        nameList = [line.split(',')[0] for line in lines]


    # If the name is not already in the attendance list, append the name and current time to the CSV file
    if stu_id not in nameList:
        with open(csv_file, 'a') as f:
            f.write(f'{stu_id},{now.strftime("%H:%M:%S")}\n')



async def main():
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

                if not matches[matchIndex] and faceDis[matchIndex] >= 0.4:
                    # If an unknown face is detected with similarity >= 0.4, print "unknown" and skip marking attendance
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
                    tasks.append(markattendanceasync(stu_id))

                    imgBackground[44:44 + 633, 800:800 + 414] = imgModeList[1]

            if tasks:
                await asyncio.gather(*tasks)

        cv2.imshow("Attendance System", imgBackground)

        # If the 'q' key is pressed, break out of the loop and stop the video capture
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

        # Release the video capture and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()


# Run the asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())