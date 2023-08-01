import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials, firestore, storage
import tempfile

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "attendance-system-0786.appspot.com"
})

db = firestore.client()
bucket = storage.bucket()


def download_images_from_storage(folder_path):
    path_list = os.listdir(folder_path)
    img_list = []
    student_ids = []

    for path in path_list:
        file_name = os.path.join(folder_path, path).replace('\\', '/')  # Use os.path.join to handle path separators
        _, temp_file_path = tempfile.mkstemp(suffix='.jpg')
        blob = bucket.blob(file_name)
        blob.download_to_filename(temp_file_path)

        img_list.append(cv2.imread(temp_file_path))
        student_ids.append(os.path.splitext(path)[0])

    return img_list, student_ids


def findencoding(images_list):
    encodeList = []
    for img in images_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


folderPath = 'DataSet'  # Assuming the folder name is "DataSet"

print('Downloading Images from Firebase Storage ...')
imgList, studentIds = download_images_from_storage(folderPath)

print('Encoding Started ...')
encodeListKnown = findencoding(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print('Encoding Complete')

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()

print("File Saved")
