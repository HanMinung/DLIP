# Load image
from module import *

img = cv.imread('../Image/Pattern_original.tif')

blur = cv.blur(img,(5,5))                   # default : box filter
gblur = cv.GaussianBlur(img,(5,5),0)        # Gaussian blur
median = cv.medianBlur(img, 5)               # Median blur

# Plot results
plt.subplot(221),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(gblur),plt.title('Gaussian Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(median),plt.title('Median')
plt.xticks([]), plt.yticks([])
plt.show()

