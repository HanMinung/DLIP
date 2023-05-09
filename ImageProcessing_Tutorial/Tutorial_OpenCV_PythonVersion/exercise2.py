# segment only blue colored ball

from module import *

def nothing(x):
    pass

# trackbar window 생성
cv.namedWindow('image')

initHmin, initSmin, initVmin = 110, 100, 100    
initHmax, initSmax, initvmax = 130, 255, 255

cv.createTrackbar('hmin', 'image', initHmin, 179, nothing)
cv.createTrackbar('hmax', 'image', initHmax, 179, nothing)
cv.createTrackbar('smin', 'image', initSmin, 255, nothing)
cv.createTrackbar('smax', 'image', initSmax, 255, nothing)
cv.createTrackbar('vmin', 'image', initVmin, 255, nothing)
cv.createTrackbar('vmax', 'image', initvmax, 255, nothing)

img = cv.imread('Image/color_ball.jpg', cv.IMREAD_COLOR)

# Convert BRG to HSV 
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# matplotlib: cvt color for display
imgPlt = cv.cvtColor(img, cv.COLOR_BGR2RGB)

while True:
    
    hmin = cv.getTrackbarPos('hmin', 'image')
    hmax = cv.getTrackbarPos('hmax', 'image')
    smin = cv.getTrackbarPos('smin', 'image')
    smax = cv.getTrackbarPos('smax', 'image')
    vmin = cv.getTrackbarPos('vmin', 'image')
    vmax = cv.getTrackbarPos('vmax', 'image')

    lower_range = np.array([hmin, smin, vmin])
    upper_range = np.array([hmax, smax, vmax])

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, lower_range, upper_range)

    cv.imshow('image', mask)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()