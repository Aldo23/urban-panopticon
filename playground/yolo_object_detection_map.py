import cv2
import imutils
import numpy as np
import copy
import transformations as transf
import mapping
from matplotlib import pyplot as plt
import pandas as pd
import folium
from pyproj import Proj, transform

# Load Yolo
# loading trained network
net = cv2.dnn.readNet("./cfg/yolov3.weights", "./cfg/yolov3.cfg")
classes = []  # classes are categories that we are detecting
with open("./cfg/coco.names", "r") as f:  # loading classes
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img_org = cv2.imread('./playground/media/img_marked.jpg')
img_org = imutils.resize(img_org, width=500)

img = copy.deepcopy(img_org)
height, width, channels = img.shape

# Loading corners location
corners_geo = pd.read_csv('./data/corners_bcn_3857.csv')
print(corners_geo)

# Detecting objects
blob = cv2.dnn.blobFromImage(
    img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
points = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and classes[class_id] == 'person':
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # lowest point of the box (on the ground)
            points.append([center_x, int(center_y+h/2)])
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label+str(i), (x, y + 30), font, 2, color, 2)

points = np.asarray(points, dtype=np.float32)
points = np.array([points])
markers = transf.find_markers(img_org)
corners, img_org = transf.sort_markers(markers, img_org)

# Warp for coordinate projection
warped, M_geo = transf.four_point_transform_geo(img_org, corners, corners_geo)
pointsOut = cv2.perspectiveTransform(points, M_geo)
pointsOut = np.asarray(pointsOut)
pointsOut = pointsOut.tolist()[0]

# Prepare dataframe 
df = pd.DataFrame(pointsOut, columns=list('XY'))

print(df)

# Convert CRS to 4326 for map projection
df_4326 = mapping.convertCRS(df, 3857, 4326)
df_4326.to_csv('data_bcn_4326.csv', index = False)

# Create basemap
m = mapping.createBaseMap(df_4326)

# Create dot map
mapping.createDotMapSimple(df_4326, m, 'orange')

# Export map to HTML
folium.Map.save(m,'map_img.html')

print('\n Map created')
