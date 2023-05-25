""""
* author   :   Min-Woong Han
* Date     :   2023 - 05 - 26
* Brief    :   DLIP LAB4 Parking management using deep learning
"""

from module import *

class DETECTION : 
    
    def __init__(self) :
        
        self.colorYellow  = (0, 255, 255)
        self.colorGreen   = (0, 255, 0)
        self.colorPurple  = (238, 130, 238)
        self.colorWhite   = (255, 255, 255)
        self.colorBlack   = (0, 0, 0)
        self.colorBlue    = (255, 0, 0)
        self.colorRed     = (0, 0, 255)
         
        self.model        = torch.hub.load('ultralytics/yolov5', 'yolov5m6', pretrained = True)
        self.confThresh   = 0.2
        self.distThresh   = 50
         
        self.roiRegion    = [[0,260], [1280,260], [1280,410], [0, 410]]
        self.frameNum     = 0
         
        self.parkingMax   = 13
        self.carCnt       = 0 
        self.centerX      = []
         
        self.timeStart    = 0
        self.timeFinal    = 0
        self.offsetThresh = 20
        
        self.userFont    = cv.FONT_HERSHEY_COMPLEX
    
    
    def getroiFrame(self, frame):
        
        mask = np.zeros_like(frame)
        cv.fillPoly(mask, np.array([self.roiRegion], dtype=np.int32), self.colorWhite)
        
        return cv.bitwise_and(frame, mask)
    
    
    def printInfo(self, frame) :
        
        cv.putText(frame, f'# of cars : {self.carCnt}', (20, 540), self.userFont, 1, self.colorWhite, 2)
        cv.putText(frame, f'# of available space : {self.parkingMax - self.carCnt}', (20, 590), self.userFont, 1, self.colorWhite, 2)
        cv.putText(frame, f'frame number : {self.frameNum}', (20, 640), self.userFont, 1, self.colorWhite, 2)
        cv.putText(frame, f'FPS : {round(1/(self.timeFinal - self.timeStart))}', (20, 690), self.userFont, 1, self.colorWhite, 2)
        
    
    
    """
        - All process is handled in method 'mainProcess'
    """
    
    def mainProcess(self):
        
        cap = cv.VideoCapture('testVideo.avi')

        fourcc   = cv.VideoWriter_fourcc(*'XVID')
        outVideo = cv.VideoWriter('DLIP_LAB_PARKING_VIDEO_21800772_한민웅.avi', fourcc, 20.0, (1280, 720))

        f = open("counting_result.txt",'w')

        if cap.isOpened() == False :
            print("Video is not loaded ...!")
            return

        while(cap.isOpened()):
            
            self.timeStart = time.time()
            
            ret, frame = cap.read()
            
            if not ret :
                print("Frame number is over ...!")
                break

            roiFrame = self.getroiFrame(frame)
            
            detResult = self.model(roiFrame)
            
            detresultPd = detResult.pandas()
            
            objectLen = len(detresultPd.xyxy[0])
            
            for Idx in range(objectLen) :
                
                xMin = round(detresultPd.xyxy[0].xmin[Idx])
                xMax = round(detresultPd.xyxy[0].xmax[Idx])
                yMin = round(detresultPd.xyxy[0].ymin[Idx])
                yMax = round(detresultPd.xyxy[0].ymax[Idx])
                
                if detresultPd.xyxy[0].confidence[Idx] > self.confThresh :
                
                    # Car, Truck, Bus ==> Car class
                    if detresultPd.xyxy[0].name[Idx] == "car" or detresultPd.xyxy[0].name[Idx] == "truck" or detresultPd.xyxy[0].name[Idx] == "bus" :

                        xCen = (xMin + xMax) / 2

                        if any([abs(xCen - prevcenX) < self.distThresh for prevcenX in self.centerX]):  
                            continue  
                        
                        self.centerX.append(xCen)

                        if (yMin + yMax)/2 < 386 : 
                         
                            self.carCnt += 1
                            cv.rectangle(frame, (xMin, yMin), (xMax, yMax), self.colorYellow, 2)
                    
            self.timeFinal = time.time()
                    
            self.printInfo(frame)
            
            f.write(f"{self.frameNum} {self.carCnt}\n")
            
            self.centerX = []
            self.carCnt = 0
            self.frameNum += 1
            
            outVideo.write(frame)
            cv.imshow('Frame', frame)

            if cv.waitKey(1) == 27 :
                break
            
            if self.frameNum > 2500 : 
                
                print("Frame number is overed ...!")
                break
                
        f.close()
        outVideo.release()
        cap.release()
        cv.destroyAllWindows()


            
if __name__ == "__main__" :
    
    detection = DETECTION()
    
    detection.mainProcess()
    