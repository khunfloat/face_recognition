import face_recognition, os, pickle, time

start = time.time()
student = dict()
for student_id in os.listdir("asset"): 
        student[student_id.split(".")[0]] = face_recognition.face_encodings(face_recognition.load_image_file("asset\\" +student_id))[0]

file = open('training','wb')
pickle.dump(student,file)
file.close()
print("Training Done in",round((time.time()-start),2),"s")
