from module import *

# Global variable
imgWidth, imgHeight = 640, 480


# Function for setting ROI region
def setROI(img):
    
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    roiVerticies = np.array([[(imgWidth*5.0/10, height/2 + 50), (imgWidth * 5.0/10, height/2 + 50), (imgWidth * 8/10, height - 70), (imgWidth * 2/10, height - 70)]], dtype=np.int32)
    cv.fillConvexPoly(mask, roiVerticies, 255)
    # cv.imshow("mask", mask)
    imgMask = cv.bitwise_and(img, mask)

    # cv.imshow("ROI result",imgMask)
    return imgMask


# Function for preprocessing
def preProcess() :
    
    frameBlur = cv.GaussianBlur(frame, (5, 5), 0)
    # cv.imshow("blur", frameBlur)
    frameGray = cv.cvtColor(frameBlur, cv.COLOR_BGR2GRAY)
    
    frameEdge = cv.Canny(frameGray, 100, 300) 
    # cv.imshow("Canny", frameEdge)
    
    return frameEdge



class processing() :
    
    def __init__(self) :
        
        # Variable initialization : about colors
        self.colorYellow = (0, 255, 255)
        self.colorGreen  = (0, 255, 0)
        self.colorPurple = (0, 50, 255)
        self.colorWhite  = (255, 255, 255)
        self.colorBlack  = (0, 0, 0)
        self.colorRed    = (0, 0, 255)
        self.leftColor   = None
        self.rightColor  = None
        
        # Variable initialization : about line detection
        self.minAccum = 10
        self.minlineLen = 5
        self.minGap  = 200
        
        # Variable initialization : about main processing part
        self.slope   = 0.0
        self.bias    = 0.0
        self.vanishX = 0.0
        self.vanishY = 0.0
        
        self.DEG2RAD       = pi/180
        self.RAD2DEG       = 180/pi
        self.leftLine      = []
        self.rightLine     = []
        self.leftLineAvg   = None
        self.rightlineAvg  = None
        self.prevLeftLine  = None 
        self.prevRightLine = None
        self.prevIntersect = None
        self.leftlineExt   = None
        self.rightlineExt  = None
        self.changeLine    = None
        self.changelineAvg = None
        self.changelineExt = None
        self.changeprevExt = None
        self.changeSlope   = 0.0
        self.warnCnt       = 0 
        self.changeCnt     = 0 
        self.laneInfo = (50, 125)
        
        self.font = cv.FONT_HERSHEY_COMPLEX
        
        self.changeWarn    = False
    
    
    # Function for calculating vanishing point
    def lineIntersection(self, line1, line2):
        
        xDiff = (line1[0] - line1[2], line2[0] - line2[2])
        yDiff = (line1[1] - line1[3], line2[1] - line2[3])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xDiff, yDiff)

        if div == 0:
            return None

        d = det((line1[0], line1[1]), (line1[2], line1[3])), det((line2[0], line2[1]), (line2[2], line2[3]))
        x = det(d, xDiff) / div
        y = det(d, yDiff) / div

        self.prevIntersect = (int(x), int(y))       # update previous vanishing point
        
        return int(x), int(y)


    # Function to extend line ( vanishing point - image end )
    def extendLine(self, line, vanishingPoint):

        if vanishingPoint is None:
            vanishingPoint = self.prevIntersect

        if line is None :
            return None
        
        x1, y1, x2, y2 = line
        self.vanishX, self.vanishY = vanishingPoint

        # Avoid division by zero
        if x2 - x1 == 0:  
            return None

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Find the point where the line intersects the bottom of the image
        y = imgHeight
        if slope == 0 :
            return None

        x = (y - intercept) / slope

        # Find the intersection point between the original line and the line through the vanishing point
        xIntersect = (self.vanishY - intercept) / slope
        yIntersect = slope * xIntersect + intercept

        # Check if any value is NaN : exception handling
        if any(isnan(val) for val in (xIntersect, yIntersect, x, y)) :  
            return None

        return int(xIntersect), int(yIntersect), int(x), int(y)

    # Function to draw extended lines
    def drawLines(self, frame, line, color) :  

        x1, y1, x2, y2 = line
        cv.line(frame, (x1, y1), (x2, y2), color, 6)   


    # Function for calculating bias
    def calcBias(self, line1, line2):

        # Bias calculation remains the same as before
        line1X = (line1[2] + (line1[0] - line1[2])//4) 
        line2X = (line2[2] - (line2[2] - line2[0])//4) 

        roadCen = (line1X + line2X) // 2
        vehicleCen = imgWidth // 2
        bias = round((vehicleCen - roadCen) / roadCen * 100.0, 2)
        
        cv.line(frame, (roadCen, imgHeight), (roadCen, imgHeight - 30), self.colorPurple, 2)             # center of detected road
        cv.line(frame, (vehicleCen, imgHeight), (vehicleCen, imgHeight - 30), self.colorWhite, 2)        # center of vehicle

        if bias < 0:
            self.biasDirection = "left"
        else:
            self.biasDirection = "right"

        cv.putText(frame, f"BIAS : {self.biasDirection} {abs(bias)}[%]", (50, 80), cv.FONT_HERSHEY_COMPLEX, 1, self.colorBlack, 2)
        return bias

    
    def checklaneChange(self) :
        
        if(abs(self.bias) > 20) :
            
            self.warnCnt += 1
            
            if self.warnCnt > 20 :
                self.changeWarn = True
                self.warnCnt = 0

        else : 
            self.warnCnt = 0
            cv.putText(frame, "In Line? : Safe", self.laneInfo, self.font, 1, self.colorGreen, 2)
            

    # Main processing part
    def postProcess(self) :
        
        self.leftLine  = []
        self.rightLine = []
        
        frameRoi = setROI(frameEdge)
        lines = cv.HoughLinesP(frameRoi, 1, np.pi / 180, self.minAccum, minLineLength = self.minlineLen, maxLineGap = self.minGap)
        
    
        if self.changeWarn is not True and lines is not None :
            
            for line in lines :
                x1, y1, x2, y2 = line[0]
                
                if x2 - x1 == 0:
                    continue
                
                self.slope = (y2 - y1)/(x2 - x1)
                
                if   self.slope > tan(-85 * self.DEG2RAD) and self.slope < tan(-30 * self.DEG2RAD): 
                    self.leftLine.append(line[0])
                
                elif self.slope < tan(85 * self.DEG2RAD) and self.slope > tan(30 * self.DEG2RAD) :
                    self.rightLine.append(line[0])
        
            if self.leftLine :
                self.leftlineAvg  = np.mean(self.leftLine, axis = 0, dtype=np.int32)
                self.prevLeftLine = self.leftlineAvg
                self.leftColor    = self.colorGreen
            
            else :
                self.leftlineAvg = self.prevLeftLine
                self.leftColor   = self.colorYellow
                
            if self.rightLine :
                self.rightlineAvg  = np.mean(self.rightLine, axis = 0, dtype=np.int32)
                self.prevRightLine = self.rightlineAvg
                self.rightColor    = self.colorGreen
            
            else :
                self.rightlineAvg = self.prevRightLine
                self.rightColor   = self.colorYellow
            
            self.intersection = self.lineIntersection(self.leftlineAvg, self.rightlineAvg)     
            cv.circle(frame, self.intersection, 10, self.colorGreen, -1) 
        
            self.leftlineExt  = self.extendLine(self.leftlineAvg, self.intersection)   if self.leftlineAvg  is not None else None
            self.rightlineExt = self.extendLine(self.rightlineAvg, self.intersection) if self.rightlineAvg is not None else None                      
            
            self.bias = self.calcBias(self.leftlineExt, self.rightlineExt)
            
            self.drawLines(frame, self.leftlineExt, self.leftColor)
            self.drawLines(frame, self.rightlineExt, self.rightColor)
            
            pts = np.array([self.leftlineExt[:2], self.leftlineExt[2:], self.rightlineExt[2:], self.rightlineExt[:2]], dtype=np.int32)
            self.prev_pts = pts
            
            self.checklaneChange()
            
            overlay = frame.copy()
            cv.fillConvexPoly(overlay, pts, self.colorGreen)
            cv.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            cv.imshow("result", frame)
            
            
        elif self.changeWarn is True:
            
            pts = np.array([[(imgWidth*4.9/10, imgHeight/2 + 40), (imgWidth * 5.1/10, imgHeight/2 + 40), (imgWidth * 8/10, imgHeight), (imgWidth * 2/10, imgHeight)]], dtype=np.int32)

            overlay = frame.copy()
            cv.fillConvexPoly(overlay, pts, self.colorRed)
            cv.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)


            cv.circle(frame, self.prevIntersect, 10, self.colorGreen, -1)
            cv.putText(frame, "In Line? : Lane Change", self.laneInfo, self.font, 1, self.colorRed, 2)
            
            if(self.biasDirection == "right") :
                frameArrow = cv.arrowedLine(frame, (imgWidth//2, imgHeight//2), (imgWidth//2+100, imgHeight//2), (0, 0, 255), 2, tipLength=0.5)
                
            elif(self.biasDirection == "left") :
                frameArrow = cv.arrowedLine(frame, (imgWidth//2, imgHeight//2), (imgWidth//2-100, imgHeight//2), (0, 0, 255), 2, tipLength=0.5)
            
            self.changeLine = []
            
            if lines is not None:
            
                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    if x2 - x1 == 0:
                        continue
                    
                    self.slope = (y2 - y1) / (x2 - x1)

                    if self.slope > tan(-80 * self.DEG2RAD) and self.slope < tan(-40 * self.DEG2RAD):
                        self.changeLine.append(line[0])

                    elif self.slope < tan(80 * self.DEG2RAD) and self.slope > tan(40 * self.DEG2RAD):
                        self.changeLine.append(line[0])


            if self.changeLine:
                
                self.changelineAvg = np.mean(self.changeLine, axis=0, dtype=np.int32)
                self.changelineExt = self.extendLine(self.changelineAvg, self.prevIntersect) if self.changelineAvg is not None else None
                self.changeprevExt = self.changelineExt
                
                if self.changeprevExt is not None : 
                    x1, y1, x2, y2 = self.changeprevExt
                    
                    if x2 - x1 != 0 :
                        self.changeSlope = (y2 - y1)/(x2 - x1)

                    else : 
                        self.changeSlope = float('inf')

                else :
                    self.changelineExt = self.changeprevExt 
    
                if self.biasDirection == "right" :
                    
                    if self.changeSlope < tan(-65 * self.DEG2RAD)  and self.changeSlope > tan(-75 * self.DEG2RAD) :
                        
                        self.changeCnt += 1
                        
                        if(self.changeCnt >= 12) : 
                            self.changeWarn = False
                            self.changeWarn = 0 
                            self.changeCnt  = 0
                        
                elif self.biasDirection == "left" :
                    
                    if self.changeSlope > tan(70 * self.DEG2RAD)  and self.changeSlope < tan(75 * self.DEG2RAD) :
                        
                        self.changeCnt += 1
                        
                        if(self.changeCnt >= 12) : 
                            self.changeWarn = False
                            self.changeWarn = 0 
                            self.changeCnt  = 0
                        
            if self.changelineExt is not None :
                self.drawLines(frame, self.changelineExt, self.colorGreen)
            
            else : 
                if self.changeprevExt is not None :
                    self.drawLines(frame, self.changeprevExt, self.colorYellow)
            
            
        
    def showFPS(self, gapTime) :
        
        FPS = round(1./gapTime)
        cv.putText(frame, f"FPS : {FPS}", (50, 170), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)       # calculation of FPS
        cv.imshow('result', frame)
        



if __name__ == "__main__" :
    
    # cap = cv.VideoCapture('../DLIP_LAB3/testset/road_straight.mp4')
    cap = cv.VideoCapture('../DLIP_LAB3/testset/road_lanechange.mp4')
    
    process = processing()

    # fourcc = cv.VideoWriter_fourcc(*'mp4v')  # mp4 형식을 사용합니다.
    # out = cv.VideoWriter('output.mp4', fourcc, 30.0, (imgWidth, imgHeight))

    while(cap.isOpened()):

        startTime = timeit.default_timer()

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv.resize(frame, (imgWidth, imgHeight))
        
        frameEdge = preProcess()
        
        process.postProcess()

        terminateTime = timeit.default_timer()

        process.showFPS(terminateTime - startTime)

        # out.write(frame)

        k = cv.waitKey(27) & 0xFF
        if k == ord('q')   :   
            break
        elif k == ord('s') :   
            cv.waitKey()

        
    cap.release()
    # out.release()
    cv.destroyAllWindows()