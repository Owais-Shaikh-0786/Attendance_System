import os
import firebase_admin
from firebase_admin import credentials, firestore, storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "attendance-system-0786.appspot.com"
})

db = firestore.client()
bucket = storage.bucket()

def upload_images_to_storage(folder_path):
    path_list = os.listdir(folder_path)

    for path in path_list:
        file_name = f'{folder_path}/{path}'
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)
        print("Image uploaded:", path)


print('Uploading Images to Firebase Storage ...')
folderPath = 'DataSet'  # Assuming the folder name is "DataSet"
upload_images_to_storage(folderPath)