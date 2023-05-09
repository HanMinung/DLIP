# Read Image
from module import *

src = cv.imread('../Image/coins.png')

# Convert it to grayscale:
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)


# Reduce noise and avoid false circle detection
gray = cv.medianBlur(gray, 5)

# Hough Circle Transform
rows = gray.shape[0]
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=30)

# Draw circle
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # Draw circle center
        cv.circle(src, center, 1, (0, 100, 100), 3)
        # Draw circle outline
        radius = i[2]
        cv.circle(src, center, radius, (255, 0, 0), 3)


# Plot images
titles = ['Original with Circle Detected']
srcPlt=cv.cvtColor(src,cv.COLOR_BGR2RGB)
plt.imshow(srcPlt)
plt.title(titles)
plt.xticks([]),plt.yticks([])
plt.show()