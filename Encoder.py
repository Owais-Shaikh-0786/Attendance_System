import cv2
import face_recognition
import pickle
import os

folderPath = 'DataSet'
pathList = os.listdir(folderPath)
print("List of Images: ", pathList)

imgList = []
studentIds = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])
    print("only id : ", os.path.splitext(path)[0])


def findencoding(images_list):
    encodeList = []
    for img in images_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print('Encoding Started ...')

encodeListKnown = findencoding(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print('Encoding Complete')

file = open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()

print("File Saved")
