from module import *

minLen, minGap = 10, 10

def resizeImg(src) :
    
    dst = cv.resize(src, dsize = (640, 480))
    
    return dst

def onTrackbar(val):
    
    minLineLen = cv.getTrackbarPos('Min Line Length', 'Line Detection')
    minLineGap = cv.getTrackbarPos('Max Line Gap', 'Line Detection')

    srcEdge = cv.Canny(src, 50, 250, None, 3)
    cv.imshow("edge detection", srcEdge)
    
    lines = cv.HoughLinesP(srcEdge, 1, np.pi / 180, 100, minLineLength = minLineLen, maxLineGap = 5)
    # lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength = minLineLen, maxLineGap = minLineGap)

    img = src.copy()

    if lines is not None :
        for line in lines :
            x1, y1, x2, y2 = line[0] 
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv.imshow('Line Detection', img)


if __name__ == "__main__" :
    
    src = cv.imread('../DLIP_LAB3/testset/straightLineTest_2.jpg', cv.IMREAD_ANYCOLOR)
    
    src = resizeImg(src)
    
    srcBlur = cv.GaussianBlur(src, (5,5), 1)
    srcGray = cv.cvtColor(srcBlur, cv.COLOR_BGR2GRAY)                    # Grayscale conversion

    cv.namedWindow('Line Detection')

    cv.createTrackbar('Min Line Length', 'Line Detection', minLen, 500, onTrackbar)
    cv.createTrackbar('Max Line Gap', 'Line Detection', minGap, 500, onTrackbar)
    
    cv.setTrackbarPos('Min Line Length', 'Line Detection', 10)
    cv.setTrackbarPos('Max Line Gap', 'Line Detection', 10)
    
    while True:

        onTrackbar(0)

        if cv.waitKey(1) & 0xFF == 27:
         break

    cv.destroyAllWindows()