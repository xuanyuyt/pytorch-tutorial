import cv2
import numpy as np

for i in range(9):
    str_idx = str(i+1)
    img_path = 'G:/test/image_example/2/{}.jpg'.format(str_idx)
    img =  cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
    x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    
    
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    
    gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    (thresh, img_bin) = cv2.threshold(gray, 20, 255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
    dilation = cv2.dilate(img_bin, kernel)
    
    
    # lines = cv2.HoughLines(dilation, 1, np.pi / 180, 118)
    result = img.copy()
    minLineLength = 10  # height/32
    maxLineGap = 50  # height/40
    lines = cv2.HoughLinesP(dilation, 1, np.pi / 180, 50, minLineLength, maxLineGap)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if (abs(x2-x1) >= img.shape[0] / 2 - 10) or (abs(y2-y1) >= img.shape[0] / 4 - 10):
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('result', result)
    
    # cv2.imshow("absX", absX)
    # cv2.imshow("absY", absY)
    # cv2.imshow("Result", img_bin)
    cv2.imshow("erosion", dilation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
