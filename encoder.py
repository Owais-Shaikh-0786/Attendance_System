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


def create_data_dictionary(image_id):
    # Prompt the user to enter details for the image
    stu_name = input(f"Enter Student Name for {image_id}: ")
    data = {
        'id': image_id,
        'name': stu_name,
        # Add more fields if needed, for example: 'path': '', 'date_taken': '', etc.
    }
    return data


def upload_data_to_firestore(data, collection_name, document_name):
    doc_ref = db.collection(collection_name).document(document_name)
    doc_ref.set(data)
    print(f"Data uploaded to Firestore for {collection_name}/{document_name}.")


def upload_images_to_storage(folder_path):
    path_list = os.listdir(folder_path)

    for path in path_list:
        image_id = os.path.splitext(path)[0]
        file_name = f'{folder_path}/{path}'

        data = create_data_dictionary(image_id)  # Create a data dictionary for the image

        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)  # Upload the image without metadata

        # Set the metadata for the blob
        blob.metadata = data
        blob.patch()

        # Upload the data to Firestore
        collection_name = "studentsDetails"  # You can change this to the desired collection name
        document_name = image_id
        upload_data_to_firestore(data, collection_name, document_name)

        print("Image and data uploaded:", path)


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


print('Uploading Images and Data to Firebase Storage and Firestore ...')
folderPath = 'dataSet'  # Assuming the folder name is "dataSet"
upload_images_to_storage(folderPath)

print('Downloading Images from Firebase Storage ...')
imgList, studentIds = download_images_from_storage(folderPath)

print('Encoding Started ...')
encodeListKnown = findencoding(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print('Encoding Complete')

file = open("encodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()

print("File Saved")
