from module import *

def mouse_click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'Clicked coordinates: X={x}, Y={y}')

cap = cv.VideoCapture('testVideo.avi')

width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
print(f'Video size: {width}x{height}')

cv.namedWindow('Video')
cv.setMouseCallback('Video', mouse_click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv.imshow('Video', frame)
    if cv.waitKey(1) == 27:  # If ESC is pressed, exit loop
        break

cap.release()
cv.destroyAllWindows()
