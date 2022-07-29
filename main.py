#importing modules
import glob
import os
import time
import cv2
import face_recognition
from simple_facerec import SimpleFacerec
from vehicle_detector import VehicleDetector

#finding objects (car, bus, truck)
class FindObjects:
    def __init__(self, keyword, path_to_videos):
        self.keyword = keyword
        self.path = path_to_videos
        self.meta = []
    def find_person(self, video):   #function to detect person images
        sfr = SimpleFacerec()
        sfr.load_encoding_images("../venv/person_images/") # GIVE YOUR FILE PATH
        cap = cv2.VideoCapture(video)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                return
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                if "Sundar" in name: # Hard code the person name you want to detect from submitted mp4 files (DATA Folder in the zip)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
                    cv2.imwrite("../venv/Detected_Image/" + str(i) + os.path.basename(video) + ".jpg", frame) # GIVE YOUR FILE PATH
                    self.meta.append(video)
                    return
            i += 1
        cap.release()
        cv2.destroyAllWindows()

    def find_objects(self, object_name, video_src): # Detecting car, bus and truck using vehicle detector method imported from vehicle_detector.py
        vd = VehicleDetector()
        success = True
        vidcap = cv2.VideoCapture(video_src)
        vidcap.set(cv2.CAP_PROP_FPS, 10)

        print(vidcap.get(cv2.CAP_PROP_FPS))
        frame_rate = 0.5
        prev = 0
        while success:
            time_elapsed = time.time() - prev
            success, image = vidcap.read()
            if time_elapsed > 1. / frame_rate:
                print(time.time())
                prev = time.time()
                vehicles_folder_count = 0
                vehicle_boxes = vd.detect_vehicles(image)
#                vehicle_count = len(vehicle_boxes)
                # Update total count
#                vehicles_folder_count += vehicle_count

                for box in vehicle_boxes:
                    x, y, w, h = box
                    print(vd.getObjectName())

                    if self.keyword in vd.getObjectName():
                        #cv2.rectangle(image, (x, y), (x + w, y + h), (25, 0, 180), 3)

                        text = '%s' % (self.keyword)
                        cv2.putText(image, text, (20, 50), 0, 2, (100, 200, 0), 3)
                        cv2.imwrite("../venv/Detected_Image/"  + os.path.basename(video_src) + ".jpg", image) # GIVE YOUR FILE PATH 

                     #   if vehicles_folder_count > 0:
                        self.meta.append(video_src)
                        return
                #uncomment to see the detected images
                # cv2.imshow(object_name, image)
                # if cv2.waitKey(1) == 13:  # is the Enter Key
                #     break

    # Checking user input
    def run(self):
        for files in os.listdir(self.path):
            path_to_video = os.path.join(self.path, files)
            if "DS_Store" in path_to_video:
                continue
            print("examining for " + path_to_video)
            if self.keyword == "person": # Check if the given name is person to detect humans in videos.
                self.find_person(path_to_video)
            else: # If not person check for the given object
                self.find_objects(self.keyword, path_to_video)
            print("Detected " + self.keyword + " at " + str(self.meta)) # Printing output

dir_path = "../venv/Data/" # GIVE YOUR FILE PATH
keyword = input("Enter the object you want to find:") # User Input
#a=keyword
FindObjects(str(keyword), dir_path).run()
