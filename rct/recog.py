import cv2
import numpy as np
import os
from objects import CentroidTracker, TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import dlib
import matplotlib
import time

class Recognition:
    def __init__(self,caffe="MobileNetSSD_deploy.caffemodel",prototxt='MobileNetSSD_deploy.prototxt',CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]):
        self.caffe = caffe
        self.prototxt = prototxt
        self.CLASSES = CLASSES

    def recog_im_dir(self,path_to_images,list_of_ignored):
        self.path_to_images = path_to_images
        self.list_of_ignored = list_of_ignored
        conf=0.4
        IGNORE=set(list_of_ignored)
        COLORS = np.random.uniform(0, 255, size=(len(set(self.CLASSES)-set(IGNORE)), 3))
        net = cv2.dnn.readNetFromCaffe(self.prototxt, self.prototxt)
        for j in os.listdir(self.path_to_images):
            image = cv2.imread(j)
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf:

                    idx = int(detections[0, 0, i, 1])
                    if self.CLASSES[idx] in IGNORE:
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(image, (startX, startY), (endX, endY),COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(image, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            cv2.imwrite('rec_'+j,image)

    def count_single_object_from_video(self,path_to_video,path_to_out,obj):
        self.path_to_video = path_to_video
        self.path_to_out = path_to_out
        conf=0.4
        sk_fr=30
        net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
        vs = cv2.VideoCapture(self.path_to_video)
        writer = None
        W = None
        H = None
        ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        trackers = []
        trackableObjects = {}
        totalFrames = 0
        #totalDown = 0
        #totalUp = 0
        total=0
        fps = FPS().start()
        while True:
            frame = vs.read()
            frame = frame[1]
            frame = imutils.resize(frame, width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(self.path_to_out, fourcc, 30,(W, H), True)
            status = "Waiting"
            rects = []
            if totalFrames % sk_fr == 0:
                
                status = "Detecting"
                trackers = []

                
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                net.setInput(blob)
                detections = net.forward()

                
                for i in np.arange(0, detections.shape[2]):
                    
                    confidence = detections[0, 0, i, 2]
                    if confidence > conf:
                        
                        idx = int(detections[0, 0, i, 1])

                        
                        if CLASSES[idx] != obj:
                            continue

                        
                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")

                        
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)

                        
                        trackers.append(tracker)

            
            else:
                
                for tracker in trackers:
                    
                    status = "Tracking"

                    
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    rects.append((startX, startY, endX, endY))
            #cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                
                to = trackableObjects.get(objectID, None)

                
                if to is None:
                    to = TrackableObject(objectID, centroid)

                
                
                else:
                    
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    
                    if not to.counted:
                        
                        if direction < 0 and centroid[1] < H // 2:
                            #totalUp += 1
                            total += 1
                            to.counted = True

                        
                        elif direction > 0 and centroid[1] > H // 2:
                            #totalDown += 1
                            total += 1
                            to.counted = True

                
                trackableObjects[objectID] = to

                
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

            
            info = [
                #("Up", totalUp),
                #("Down", totalDown),
                ('Total',total),
                ("Status", status),
            ]

            
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                #cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            
            if writer is not None:
                writer.write(frame)

            
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            
            totalFrames += 1
            fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        
        if writer is not None:
            writer.release()

        
        else:
            vs.release()

        cv2.destroyAllWindows()





