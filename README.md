# Attendance System with Real-Time Tracking using Face Recognition
- This project implements an intelligent attendance system that leverages the power of face recognition technology to efficiently track student attendance. It uses computer vision techniques to identify students' faces in real-time through a webcam feed and integrates seamlessly with Firebase to provide real-time data tracking and retrieval capabilities.

## Features
- Real-time face detection and recognition using webcam feed.
- Automatic marking of attendance for recognized students.
- Integration with Firebase for real-time data tracking and retrieval.
- Asynchronous processing for efficient attendance marking.

## How It Works
- **Face Recognition and Attendance Marking:** The system captures real-time video feed from a webcam and employs the face_recognition library to detect and recognize students' faces. Upon successful recognition, the system marks the attendance of the identified students.
- **Firebase Integration:** Student images and details are uploaded to Firebase Cloud Storage and Firestore. Face encodings are generated and stored alongside student IDs in the encodeFile.p file.
- **Real-Time Data Tracking:** The attendance status of recognized students is dynamically updated and synchronized with Firebase Realtime Database. This ensures that the attendance data is available in real-time for further analysis and reporting.
- **Attendance Status and Time Tracking:** The system keeps track of attendance status ("Present" or "Absent") based on user-defined class start time and attendance cutoff time. The timestamp of attendance is also recorded.

## Storage 

![image](https://github.com/Owais-Shaikh-0786/Attendance_System/assets/139638554/6d94e548-43bc-4f44-bfd8-5260bead90e8)

## Cloud Firestore ( Student Database )

![image](https://github.com/Owais-Shaikh-0786/Attendance_System/assets/139638554/98e7ce33-9dee-475c-8663-c93c70d93d75)

## Cloud Firestore ( Attendance Database )

![image](https://github.com/Owais-Shaikh-0786/Attendance_System/assets/139638554/da4070d8-5824-431f-8b1f-55c0792924ab)
![image](https://github.com/Owais-Shaikh-0786/Attendance_System/assets/139638554/de729695-5bb8-4eaa-9ca0-c13be9cbb8d3)
![image](https://github.com/Owais-Shaikh-0786/Attendance_System/assets/139638554/760e613f-c92b-4357-bc35-e3930dff7e59)

