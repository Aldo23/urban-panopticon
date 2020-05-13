

"""
Course:  Training YOLO v3 for Objects Detection with Custom Data

Section-2
Objects Detection on Video with YOLO v3 and OpenCV
File: yolo-3-video.py
"""


# Detecting Objects on Video with OpenCV deep learning library
#
# Algorithm:
# Reading input video --> Loading YOLO v3 Network -->
# --> Reading frames in the loop --> Getting blob from the frame -->
# --> Implementing Forward Pass --> Getting Bounding Boxes -->
# --> Non-maximum Suppression --> Drawing Bounding Boxes with Labels -->
# --> Writing processed frames
#
# Result:
# New video file with Detected Objects, Bounding Boxes and Labels


# Importing needed libraries
import numpy as np
import cv2
import time
import transformations as transf
import mapping
import imutils
import pandas as pd
import folium
from folium.plugins import FeatureGroupSubGroup, HeatMap, HeatMapWithTime



"""
Start of:
Reading input video
"""

# Defining 'VideoCapture' object
# and reading video from a file
# Pay attention! If you're using Windows, the path might looks like:
# r'videos\traffic-cars.mp4'
# or:
# 'videos\\traffic-cars.mp4'
video = cv2.VideoCapture('./playground/media/video_test_long.mp4')

# Import .csv of location coordinates in EPSG 3857
corners_geo = pd.read_csv('./data/corners_3857.csv')

# Preparing variable for writer
# that we will use to write processed frames
writer = None

# Preparing variables for spatial dimensions of the frames
h, w = None, None

"""
End of:
Reading input video
"""
img_org = cv2.imread('./playground/media/video_corners.jpg')
markers = transf.find_markers(img_org)
corners, img_org = transf.sort_markers(markers, img_org)


"""
Start of:
Loading YOLO v3 network
"""

# Loading COCO class labels from file
# Opening file
# Pay attention! If you're using Windows, yours path might looks like:
# r'yolo-coco-data\coco.names'
# or:
# 'yolo-coco-data\\coco.names'
with open('./cfg/coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]


# # Check point
# print('List with labels names:')
# print(labels)

# Loading trained YOLO v3 Objects Detector
# with the help of 'dnn' library from OpenCV
# Pay attention! If you're using Windows, yours paths might look like:
# r'yolo-coco-data\yolov3.cfg'
# r'yolo-coco-data\yolov3.weights'
# or:
# 'yolo-coco-data\\yolov3.cfg'
# 'yolo-coco-data\\yolov3.weights'
network = cv2.dnn.readNetFromDarknet('./cfg/yolov3.cfg',
                                     './cfg/yolov3.weights')

# Getting list with names of all layers from YOLO v3 network
layers_names_all = network.getLayerNames()

# # Check point
# print()
# print(layers_names_all)

# Getting only output layers' names that we need from YOLO v3 algorithm
# with function that returns indexes of layers with unconnected outputs
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# # Check point
# print()
# print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes
# with non-maximum suppression
threshold = 0.3

# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# # Check point
# print()
# print(type(colours))  # <class 'numpy.ndarray'>
# print(colours.shape)  # (80, 3)
# print(colours[0])  # [172  10 127]

"""
End of:
Loading YOLO v3 network
"""

"""
Start of:
Reading frames in the loop
"""

# Initialize dataframe for output data
df = pd.DataFrame(columns=list('XYtf'))

# Defining variable for counting frames
# At the end we will show total amount of processed frames
f = 0

# Defining variable for counting total time
# At the end we will show time spent for processing all frames
t = 0
iteration = 0

# Defining loop for catching frames
while True:    
    iteration += 1
    # Capturing frame-by-frame
    ret, frame = video.read()

    # If the frame was not retrieved
    # e.g.: at the end of the video,
    # then we break the loop
    if not ret:
        break
    
    if iteration % 200 == 0:

        # Getting spatial dimensions of the frame
        # we do it only once from the very beginning
        # all other frames have the same dimension
        if w is None or h is None:
            # Slicing from tuple only first two elements
            h, w = frame.shape[:2]

        """
        Start of:
        Getting blob from current frame
        """

        # Getting blob from current frame
        # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob from current
        # frame after mean subtraction, normalizing, and RB channels swapping
        # Resulted shape has number of frames, number of channels, width and height
        # E.G.:
        # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)

        """
        End of:
        Getting blob from current frame
        """

        """
        Start of:
        Implementing Forward pass
        """

        # Implementing forward pass with our blob and only through output layers
        # Calculating at the same time, needed time for forward pass
        network.setInput(blob)  # setting blob as input to the network
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()

        # Increasing counters for frames and total time
        f += 10
        t += end - start

        # Showing spent time for single current frame
        print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

        """
        End of:
        Implementing Forward pass
        """

        """
        Start of:
        Getting bounding boxes
        """

        # Preparing lists for detected bounding boxes,
        # obtained confidences and class's number
        bounding_boxes = []
        confidences = []
        class_numbers = []
        class_type = []
        points = []

        # Going through all output layers after feed forward pass
        for result in output_from_network:
            # Going through all detections from current output layer
            for detected_objects in result:
                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
                # Getting value of probability for defined class
                confidence_current = scores[class_current]

                # # Check point
                # # Every 'detected_objects' numpy array has first 4 numbers with
                # # bounding box coordinates and rest 80 with probabilities
                #  # for every class
                # print(detected_objects.shape)  # (85,)

                # Eliminating weak predictions with minimum probability
                if confidence_current > probability_minimum:
                    # Scaling bounding box coordinates to the initial frame size
                    # YOLO data format keeps coordinates for center of bounding box
                    # and its current width and height
                    # That is why we can just multiply them elementwise
                    # to the width and height
                    # of the original frame and in this way get coordinates for center
                    # of bounding box, its width and height for original frame
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Now, from YOLO data format, we can get top left corner coordinates
                    # that are x_min and y_min
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min,
                                        int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)
                    class_type.append(labels[class_current])
                    points.append([x_center, int(y_center+box_height/2)])

        """
        End of:
        Getting bounding boxes
        """

        """
        Start of:
        Non-maximum suppression
        """

        # Implementing non-maximum suppression of given bounding boxes
        # With this technique we exclude some of bounding boxes if their
        # corresponding confidences are low or there is another
        # bounding box for this region with higher confidence

        # It is needed to make sure that data type of the boxes is 'int'
        # and data type of the confidences is 'float'
        # https://github.com/opencv/opencv/issues/12789
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                probability_minimum, threshold)

        """
        End of:
        Non-maximum suppression
        """
 
        """
        Start of:
        Data extraction
        """
        # Confirm data structure for points
        points = np.asarray(points, dtype=np.float32)
        points = np.array([points])
        
        # Warp for coordinate projection
        warped, M_geo = transf.four_point_transform_geo(frame, corners, corners_geo)
        pointsOut = cv2.perspectiveTransform(points, M_geo)
        pointsOut = pointsOut.tolist()[0]
        
        # Create dataframe for frame and append it
        i_df = mapping.adjustDataFrame(df, pointsOut, class_type, iteration)
        df = df.append(i_df)

        """
        End of:
        Data extraction
        """
 
    else:
        continue

"""
End of:
Reading frames in the loop
"""
# Printing final results
print()
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))


"""
Start of:
Mapping
"""

# Convert CRS to 4326 for map projection
print('\n Converting to EPSG 4326')

df_4326 = mapping.convertCRS(df, 3857, 4326)
#df_4326.to_csv('data_4326.csv', index = False)

print('\n Data exported')

# Create basemap
m = mapping.createBaseMap(df_4326)
gradient = folium.branca.colormap.linear.OrRd_04

# Create dot map
mapping.createDotMapFrame(df_4326, m, gradient)

# Export map to HTML
folium.Map.save(m,'map.html')

print('\n Map created')


"""
End of:
Mapping
"""


"""
Some comments

What is a FOURCC?
    FOURCC is short for "four character code" - an identifier for a video codec,
    compression format, colour or pixel format used in media files.
    http://www.fourcc.org


Parameters for cv2.VideoWriter():
    filename - Name of the output video file.
    fourcc - 4-character code of codec used to compress the frames.
    fps	- Frame rate of the created video.
    frameSize - Size of the video frames.
    isColor	- If it True, the encoder will expect and encode colour frames.
"""
