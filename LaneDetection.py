import cv2
import numpy as np

# vidcap = cv2.VideoCapture("LaneVideo.mp4")
vidcap = cv2.VideoCapture("test_two.mp4")
success, image = vidcap.read()

def detection_direction(lines, image):
    contador_izquierda = 0
    contador_derecha = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            pendiente = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
            if pendiente < 0:
                contador_izquierda += 1
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)  # Azul para líneas hacia la izquierda
            elif pendiente > 0:
                contador_derecha += 1
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Rojo para líneas hacia la derecha
    if contador_izquierda > contador_derecha:
        print(f"gire a la izquierda - {contador_izquierda}")
    elif contador_derecha > contador_izquierda:
        print(f"gire a la derecha - {contador_derecha}")
    else:
        print(f"siga derecho - {contador_izquierda} - {contador_derecha}")

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Threshold", "Trackbars", 50, 200, nothing)
cv2.createTrackbar("Min Line Length", "Trackbars", 40, 100, nothing)
cv2.createTrackbar("Max Line Gap", "Trackbars", 20, 100, nothing)


# cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
# cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
# cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

while success:
    success, image = vidcap.read()
    frame = cv2.resize(image, (640,480))

    # TEST ****************************************************
    # detection of lanes to courve the road
    blurred_frame = cv2.GaussianBlur(image, (5, 5), 0)

    edges = cv2.Canny(blurred_frame, 50, 150)


    threshold = cv2.getTrackbarPos("Threshold", "Trackbars")
    minLineLength = cv2.getTrackbarPos("Min Line Length", "Trackbars")
    maxLineGap = cv2.getTrackbarPos("Max Line Gap", "Trackbars")

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)


    detection_direction(lines, image)

    # contador_izquierda = 0
    # contador_derecha = 0

    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         pendiente = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
    #         if pendiente < 0:
    #             contador_izquierda += 1
    #             cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 5)  # Azul para líneas hacia la izquierda
    #         elif pendiente > 0:
    #             contador_derecha += 1
    #             cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Rojo para líneas hacia la derecha

    # if contador_izquierda > 0:
    #     print("gire a la izquierda")
    # elif contador_derecha < 0:
    #     print("gire a la derecha")
    # else:
    #     print("siga derecho")
    
    # END TEST ****************************************************


    ## Choosing points for perspective transformation
    tl = (222,387)
    bl = (70 ,472)
    tr = (400,380)
    br = (538,472)

    cv2.circle(frame, tl, 5, (0,0,255), -1)
    cv2.circle(frame, bl, 5, (0,0,255), -1)
    cv2.circle(frame, tr, 5, (0,0,255), -1)
    cv2.circle(frame, br, 5, (0,0,255), -1)

    ## Aplying perspective transformation
    pts1 = np.float32([tl, bl, tr, br]) 
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]]) 
    
    # Matrix to warp the image for birdseye window
    matrix = cv2.getPerspectiveTransform(pts1, pts2) 
    transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))

    ### Object Detection
    # Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower = np.array([l_h,l_s,l_v])
    upper = np.array([u_h,u_s,u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    #Histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    #Sliding Window
    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    while y>0:
        ## Left threshold
        img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(left_base-50 + cx)
                left_base = left_base-50 + cx
        
        ## Right threshold
        img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(right_base-50 + cx)
                right_base = right_base-50 + cx
        
        cv2.rectangle(msk, (left_base-50,y), (left_base+50,y-40), (255,255,255), 2)
        cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-40), (255,255,255), 2)
        y -= 40
        
    cv2.imshow("Original", frame)
    # cv2.imshow("Bird's Eye View", transformed_frame)
    # cv2.imshow("Lane Detection - Image Thresholding", mask)
    # cv2.imshow("Lane Detection - Sliding Windows", msk)

    if cv2.waitKey(10) == 27:
        break