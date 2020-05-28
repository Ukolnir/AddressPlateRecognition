import cv2
import matplotlib.pyplot as plt
import numpy as np

# https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
# https://www.learnopencv.com/deep-learning-based-text-detection-using-opencv-c-python/

def get_blob(image):
    mean = (123.68, 116.78, 103.94)
    blob = cv2.dnn.blobFromImage(image, 1.0, (image.shape[0], image.shape[1]), mean, True, False)
    return blob

# https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        
        # loop over all indexes in the indexes list
        for pos in range(0,last):
            # grab the current index
            j=idxs[pos]
            
            # find the largest (x,y) coords for the start of the
            # bounding box and the smallest (x,y) coords for the end
            # of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = max(x2[i], x2[j])
            yy2 = max(y2[i], y2[j])
            
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            
            # compute the ratio of overlap between the computed bounding
            # box and the bounding box in the area list
            overlap = float(w*h) / area[j]
            
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
        
        # delete all indexes from the index list 
        # that are in the suppression list
        idxs = np.delete(idxs, suppress)
    return boxes[pick]

def get_net_output(net, blob):
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")
    net.setInput(blob)
    return net.forward(outputLayers) # (scores, geometry)

def get_boxes(scores, geometry, minConfidence = 0.5, overlapThresh = 0.01):
    (numRows, numCols) = scores.shape[2:4]
    # stores the bounding box (x, y)-coordinates for text regions
    rects = [] 
    
    # stores the probability associated with each of the bounding boxes in rects
    confidences = [] 
    
    res_boxes = []
    
    for y in range(0,numRows):
        scoresData = scores[0,0,y]
        xData0 = geometry[0,0,y]
        xData1 = geometry[0,1,y]
        xData2 = geometry[0,2,y]
        xData3 = geometry[0,3,y]
        anglesData = geometry[0,4,y]
       
        for x in range(numCols):          
            # if score does not have sufficient probability, ignore it
            if scoresData[x] < minConfidence:
                continue
                
            # compute the offset factor as our resulting 
            # feature maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and 
            # height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)
            # coordinates for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            # add the bounding box coordinates and 
            # probability score to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, 
        # overlapping bounding boxes
        boxes= non_max_suppression(np.array(rects), overlapThresh)
        res_boxes.extend(boxes)
    res_boxes= non_max_suppression(np.array(res_boxes), overlapThresh)
    return res_boxes

def draw_boxes(image, boxes, rW = 1, rH = 1, color = (0,255,0), thickness=2):
    with_boxes = image.copy()    
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based 
        # on the respective ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        #draw the bounding box on the image
        cv2.rectangle(with_boxes, (startX, startY), (endX, endY), color, thickness)
    return with_boxes

def scale_boxes(boxes, rW, rH):
    new_boxes = boxes.copy()
    for box in new_boxes:
        box[0] *= rW
        box[1] *= rH
        box[2] *= rW
        box[3] *= rH
    return new_boxes

def enlarge_boxes(boxes, imgSize, scaleX, scaleY):
    new_boxes = boxes.copy()
    for box in new_boxes:
        box[0] = max(box[0] - scaleX, 0)
        box[1] = max(box[1] - scaleY, 0)
        box[2] = min(box[2] + scaleX, imgSize[0])
        box[3] = min(box[3] + scaleY, imgSize[1])
    return new_boxes
