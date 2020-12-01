import face_recognition, os, pickle, cv2, time

percentage = 60

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

file = open('training','rb')
new_dict = pickle.load(file)
file.close()

face_dist = dict()
fd = []

start = time.time()

if face_recognition.face_locations(frame) != []:
    #unknown_Pic = face_recognition.load_image_file("test\\atom.jpg")
    unknown_face_encoding = face_recognition.face_encodings(frame)[0]

    for key,value in new_dict.items():
        face_dist[(face_recognition.face_distance([value], unknown_face_encoding))[0]] = key
        fd.append((face_recognition.face_distance([value], unknown_face_encoding))[0])

    if min(fd) < (100-percentage)/100: print(face_dist[min(fd)], round(((1-min(fd))*100),2),"%")
    else: 
        print("unknow")
        print(face_dist[min(fd)], round(((1-min(fd))*100),2),"%")

    print("time use",time.time()-start,"second")

else: print("can't detect face")