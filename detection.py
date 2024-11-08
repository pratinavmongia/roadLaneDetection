import numpy as np
import pandas as pd
import cv2

from moviepy import editor
import moviepy

def drawLines(img, lines, color=[255, 0, 0], thickness=12):

    # Drawing lines onto the input image.
    line_image = np.zeros_like(img)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    return cv2.addWeighted(img, 1.0, line_image, 1.0, 0.0)

def pixelPoints(y1, y2, line):
    # Converting the slope and intercept of each line into pixel points.
    
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def avgSlopeIntercept(lines):
    # finding slope and intercept of the left and right lanes of each image.
    
    leftLines    = [] 
    leftWeights  = [] 
    rightLines   = [] 
    rightWeights = [] 
     
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            
            if slope < 0:
                leftLines.append((slope, intercept))
                leftWeights.append((length))
            else:
                rightLines.append((slope, intercept))
                rightWeights.append((length))
    leftLane  = np.dot(leftWeights,  leftLines) / np.sum(leftWeights)  if len(leftWeights) > 0 else None
    rightLane = np.dot(rightWeights, rightLines) / np.sum(rightWeights) if len(rightWeights) > 0 else None
    return leftLane, rightLane

def laneLines(img, lines):
    # full length lines from pixel points.
    
    left_lane, right_lane = avgSlopeIntercept(lines)
    y1 = img.shape[0]
    y2 = y1 * 0.6
    leftLine  = pixelPoints(y1, y2, left_lane)
    rightLine = pixelPoints(y1, y2, right_lane)
    return leftLine, rightLine
 
def houghTransform(image):

    # identifying straight lines using probabilistic Hough transform
    
    rho = 1             
    theta = np.pi/180   
    threshold = 20      
    minLineLength = 20  
    maxLineGap = 500    
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)

def regionSelection(img):
    #identifying region of interest
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    rows, cols = img.shape[:2]
    bottomLeft  = [cols * 0.1, rows * 0.95]
    topLeft     = [cols * 0.4, rows * 0.6]
    bottomRight = [cols * 0.9, rows * 0.95]
    topRight    = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottomLeft, topLeft, topRight, bottomRight]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    maskedImg = cv2.bitwise_and(img, mask)
    return maskedImg

def frameProcess(roadImg):
    # Processing the input frame to detect lane lines.
    grayscale = cv2.cvtColor(roadImg, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    low_t = 50
    high_t = 150
    edges = cv2.Canny(blur, low_t, high_t)
    region = regionSelection(edges)
    hough = houghTransform(region)
    res = drawLines(roadImg, laneLines(roadImg, hough))
    return res

def process_video(test_video, output_video):
 
    # Use the input video stream and save the output video
    
    # reading the input video file 
    input_video = editor.VideoFileClip(test_video, audio=False)
    #process the input video to detect road lines
    processedVid = input_video.fl_image(frameProcess)
    # save the output video stream to an mp4 file
    processedVid.ipython_display()
    processedVid.write_videofile(output_video, audio=False)
    

process_video("roadVid.mp4","output.mp4")