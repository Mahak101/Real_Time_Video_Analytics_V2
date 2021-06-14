#Authors - Divyam verma, Devansh Messon
#Added comments for explanations


#Importing dependencies
import torch
import cv2

#Video to be analyzed
video="traffic.mp4"

#creating object for capturing the video in order to select ROI
cap = cv2.VideoCapture(video)

#Loop for selecting Region of interest
while True:
    ret, frame = cap.read()
    x1, y1, x2, y2 = cv2.selectROI(frame)
    x3=x2+x1
    y3=y2+y1
    roi = frame[y1:y3, x1:x3]
    break
#ROI selected!

#creating object for capturing the video in order to analyze the video
cap = cv2.VideoCapture(video)

#Loading pre-trained YOLOv5 model from Pytorch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#Loop for traversing the video frame by frame
while True:
    status, frame = cap.read()
    if status==0:
        break
    #Taking out ROI from frame in another image called "roi"
    roi = frame[y1:y3,x1:x3]

    #Converting roi to list
    roi= [roi]

    #Pushing this list into YOLOv5 model to classify objects
    results = model(roi)

    #Updating the roi by classifying the objects in the frame and assigning it to results.imgs
    results.render()  # updates results.imgs with boxes and labels


    height, width, _ = results.imgs[0].shape  #can be written as height=y2-y1, width=x2-x1


    #Parameter values for making rectangle in order to show ROI on frame
    tf_x = 0
    tf_y = 0
    bt_x = width
    bt_y = height
    rect_color = (100, 0, 100)
    rect_thick = 7

    #Making rectange on the ROI
    cv2.rectangle(results.imgs[0], (tf_x, tf_y), (bt_x, bt_y), rect_color, rect_thick)

    #Assigning the update ROI to the frame to be displayed
    frame[y1:y3, x1:x3] = results.imgs[0]

    #Displaying the frame
    cv2.imshow("YOLO FRAME",frame)

    #Running the video at 1 frame per 1 millisecond
    key = cv2.waitKey(1)
    #Press "a" to exit the video
    if key == ord("a"):
        break

cap.release()
cv2.destroyAllWindows()
