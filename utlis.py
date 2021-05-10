import cv2
import numpy as np

def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def rectContours(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        if area>50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.05*peri,True)
            if len(approx)>1:
                rectCon.append(i)
    rectCon = sorted(rectCon,key=cv2.contourArea,reverse=True)
    return rectCon

def rectContours1(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        if area>50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.05*peri,True)
            if len(approx)>3:
                rectCon.append(i)
    rectCon = sorted(rectCon,key=cv2.contourArea,reverse=True)
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02* peri, True)
    return approx

def reorder(myPoints,length):
    myPoints = myPoints.reshape((length,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    myPointsNew[0]= myPoints[np.argmin(add)] # [0,0]
    myPointsNew[3]= myPoints[np.argmax(add)] # [w,h]

    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # [w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)]  # [h,0]

    return myPointsNew

def reorder2(myPoints,length):
    myPoints = myPoints.reshape((length,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[4]
    myPointsNew[3] = myPoints[2]
    myPointsNew[1] = myPoints[5]
    myPointsNew[2] = myPoints[3]

    return myPointsNew

def reorder3(myPoints,length):
    myPoints = myPoints.reshape((length,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    myPointsNew[0] = myPoints[0]
    myPointsNew[3] = myPoints[3]
    myPointsNew[1] = myPoints[4]
    myPointsNew[2] = myPoints[1]

    return myPointsNew

def splitBoxes1(img):
    y = 100
    x = 10
    h = 600
    w = 780

    y1 = 0
    x1 = 10
    h1 = 100
    w1 = 775

    y2 = 0
    x2 = 35
    h2 = 100
    w2 = 160

    crop = img[y:y + h, x:x + w]

    cols = np.hsplit(crop, 4)
    boxes = []
    for c in cols:
       rows = np.vsplit(c,10)
       for r in rows:
           crop1 = r[y2:y2 + h2, x2:x2 + w2]
           col= np.hsplit(crop1,4)
           for i in col:
               boxes.append(i)
    return boxes

def splitBoxesMSSV(img):
    cols = np.hsplit(img,6)
    boxes = []
    for c in cols:
       rows = np.vsplit(c,10)
       for r in rows:
            boxes.append(r)
    return boxes

def splitBoxesMD(img):
    cols = np.hsplit(img,3)
    boxes = []
    for c in cols:
       rows = np.vsplit(c,10)
       for r in rows:
            boxes.append(r)
    return boxes

def getIndexVal(boxes,row,col):
    # GETTING NO ZERO PIXEL VALUES OF EACH BOX
    myPixelVal = np.zeros((row, col))
    countC = 0
    countR = 0

    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if (countC == col):
            countR += 1
            countC = 0

    # FINDING INDEX VALUES OF MARKINGS
    myIndex = []
    for x in range(0, row):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
    return myIndex

def getIndexValMSSV(boxes,row,col):
    # GETTING NO ZERO PIXEL VALUES OF EACH BOX
    myPixelValMSSV = np.zeros((col, row))
    countMSSVC = 0
    countMSSVR = 0
    for image in boxes:
        totalPixelsMSSV = cv2.countNonZero(image)
        myPixelValMSSV[countMSSVC][countMSSVR] = totalPixelsMSSV
        countMSSVR += 1
        if (countMSSVR == row):
            countMSSVC += 1
            countMSSVR = 0
    myIndexMSSV = []
    for x in range(0, col):
        arr = myPixelValMSSV[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndexMSSV.append(myIndexVal[0][0])
    return myIndexMSSV

def getGrading(myIndex,ans,grading):
    #grading = []
    for x in range(0, len(ans)):
        if ans[x] == myIndex[x]:
            grading.append(1)
        else:
            grading.append(0)
    score = (sum(grading) / 5) * 10
    return score
def showAnswer(img,myIndex,grading,ans,questions,choices):
    # secW = int(img.shape[1]/questions)
    # secH = int(img.shape[0]/choices)
    secW = 60
    secH = 40
    for x in range(0,questions):
        myAns = myIndex[x]
        cX = (myAns*secW)+secW//2
        cY = (x*secH) + secH//2

        if grading[x] ==1:
            myColor = (0,255,0)
        else:
            myColor =(0,0,255)
            correctAns = ans[x]
            cv2.circle(img,((correctAns*secW)+secW//2,(x*secH)+secH//2) , 20, (0,255,0), cv2.FILLED)

        cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)
    return img