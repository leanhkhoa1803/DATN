import cv2
import numpy as np
import utlis
path = "PTN.jpg"
widthImg = 700
heightImg = 700
widthImgP2 = 800
heightImgP2 = 800
questions = 40
choices = 4
ans = [0,1,0,1,3]
mssvR=10
mssvC=6
mdR=10
mdC=3
webcamFeed = False

cameraNo=0

cap = cv2.VideoCapture(cameraNo)
cap.set(10,150)

while True:
    if webcamFeed:success,img = cap.read()
    else : img = cv2.imread(path)

    #PREPROCESSING
    img = cv2.resize(img,(widthImg,heightImg))
    img2 = cv2.resize(img,(widthImg,heightImg))

    imgContours = img.copy()
    imgFinal = img.copy()

    imgBiggestContours = img.copy()
    imgGradeContours = img.copy()

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,threshold1=10,threshold2=150)
    try:
        #FIND ALL CONTOURS
        contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours,contours,-1,(0,255,0),1)
        #FIND RECTANGLES
        rectCon = utlis.rectContours(contours)
        biggestContours = utlis.getCornerPoints(rectCon[0])

        if biggestContours.size != 0 :
            cv2.drawContours(imgBiggestContours,biggestContours,-1,(0,255,0),10)

            biggestContours= utlis.reorder(biggestContours, len(biggestContours))
            pt1 = np.float32(biggestContours)
            pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            matrix = cv2.getPerspectiveTransform(pt1,pt2)
            imgWarpColored = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

            imgContoursPart2 = imgWarpColored.copy()
            imgWarpColoredPart2 = imgWarpColored.copy()
            imgWarpColoredPart3 = imgWarpColored.copy()
            imgGray2 = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgBlur2 = cv2.GaussianBlur(imgGray2, (5, 5), 1)
            imgCanny2 = cv2.Canny(imgBlur2, threshold1=10, threshold2=150)

            contoursPart2, hierarchy = cv2.findContours(imgCanny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rectCon2 = utlis.rectContours(contoursPart2)
            biggestContours2 = utlis.getCornerPoints(rectCon2[0])
            gradePoints2 = utlis.getCornerPoints(rectCon2[0])

            cv2.drawContours(imgContoursPart2, biggestContours2, -1, (0, 255, 0), 10)
            biggestContours2 = utlis.reorder2(biggestContours2, len(biggestContours2))

            pt1Part2 = np.float32(biggestContours2)
            pt2Part2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrixPart2 = cv2.getPerspectiveTransform(pt1Part2, pt2Part2)
            imgWarpColoredPart2 = cv2.warpPerspective(imgWarpColoredPart2, matrixPart2, (widthImg, heightImg))

            gradePoints2 = utlis.reorder3(gradePoints2, len(gradePoints2))
            pt1Part3 = np.float32(gradePoints2)
            pt2Part3 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrixPart3 = cv2.getPerspectiveTransform(pt1Part3, pt2Part3)
            imgWarpColoredPart3 = cv2.warpPerspective(imgWarpColoredPart3, matrixPart3, (widthImg, heightImg))
            # #MSSV
            imgGray3 = cv2.cvtColor(imgWarpColoredPart3, cv2.COLOR_BGR2GRAY)
            imgBlur3 = cv2.GaussianBlur(imgGray3, (11, 11), 1)
            imgCanny3 = cv2.Canny(imgBlur3, threshold1=100, threshold2=200)

            imgContoursMSSV = imgWarpColoredPart3.copy()
            contoursPart3, hierarchy = cv2.findContours(imgCanny3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rectCon3 = utlis.rectContours1(contoursPart3)
            biggestContours3 = utlis.getCornerPoints(rectCon3[0])
            cv2.drawContours(imgContoursMSSV, biggestContours2, -1, (0, 255, 0), 10)

            imgWarpColoredPart4 = imgContoursMSSV.copy()
            #biggestContoursMSSV = utlis.reorder(biggestContours3, len(biggestContours3))
            biggestContoursMSSV = [[345,82],[670,81],[341,614],[672,622]]
            pt1Part4 = np.float32(biggestContoursMSSV)
            pt2Part4 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrixPart4 = cv2.getPerspectiveTransform(pt1Part4, pt2Part4)
            imgWarpColoredMSSV = cv2.warpPerspective(imgWarpColoredPart4, matrixPart4, (widthImg, heightImg))

            #MA DE
            imgWarpColoredMD = imgContoursMSSV.copy()
            # biggestContoursMSSV = utlis.reorder(biggestContours3, len(biggestContours3))
            biggestContoursMD = [[68, 84], [237, 88], [71, 616], [243, 645]]
            pt1MD = np.float32(biggestContoursMD)
            pt2MD = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrixMD = cv2.getPerspectiveTransform(pt1MD, pt2MD)
            imgWarpColoredMD = cv2.warpPerspective(imgWarpColoredMD, matrixMD, (widthImg, heightImg))

            imgWarpColoredPart2 = cv2.resize(imgWarpColoredPart2, (widthImgP2, heightImgP2))
            imgWarpColoredMSSV = cv2.resize(imgWarpColoredMSSV, (600, 600))
            imgWarpColoredMD = cv2.resize(imgWarpColoredMD, (300, 600))

            y = 100
            x = 10
            h = 600
            w = 780
            crop = imgWarpColoredPart2[y:y + h, x:x + w]

            #APPLY THRESHOLD
            imgWarpGray = cv2.cvtColor(imgWarpColoredPart2,cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray,150,255,cv2.THRESH_BINARY_INV)[1]

            boxes = utlis.splitBoxes1(imgThresh)

            #MSSV
            imgWarpGrayMSSV = cv2.cvtColor(imgWarpColoredMSSV, cv2.COLOR_BGR2GRAY)
            imgThreshMSSV = cv2.threshold(imgWarpGrayMSSV, 150, 255, cv2.THRESH_BINARY_INV)[1]
            boxesMSSV = utlis.splitBoxesMSSV(imgThreshMSSV)

            #MD
            imgWarpGrayMD = cv2.cvtColor(imgWarpColoredMD, cv2.COLOR_BGR2GRAY)
            imgThreshMD = cv2.threshold(imgWarpGrayMD, 150, 255, cv2.THRESH_BINARY_INV)[1]
            boxesMD = utlis.splitBoxesMD(imgThreshMD)

            myIndex = utlis.getIndexVal(boxes, questions, choices)

            #GRADING
            grading =[]
            score = utlis.getGrading(myIndex, ans, grading)
            #MSSV
            myPixelValMSSV = utlis.getIndexValMSSV(boxesMSSV, mssvR, mssvC)
            # MD
            myPixelValMD = utlis.getIndexValMSSV(boxesMD, mdR, mdC)
            #DISPLAY ANSWER

            imgRawGrade = np.zeros_like(img)
            cv2.putText(imgRawGrade,str(int(score)),(50,100),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,255),3)

            # invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            # imgInvWarpGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

            # imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
            # imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarpGradeDisplay,1,0)


        imgBlank = np.zeros_like(img)
        imgArray = (
                    [imgContours,imgBiggestContours,imgWarpColored,imgThresh],
                    [imgResult, imgRawDrawing, imgInvWarp, imgFinal])
    except:
        imgBlank = np.zeros_like(img)
        imgArray = ([img, imgGray, imgCanny,imgBlank],
                    [imgBlank, imgBlank, imgBlank, imgBlank],
                    [imgBlank, imgBlank, imgBlank, imgBlank])
    imgStacked = utlis.stackImages(imgArray, 0.4)

    #cv2.imshow("imgStacked",imgStacked)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Final Result.jpg",imgFinal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

