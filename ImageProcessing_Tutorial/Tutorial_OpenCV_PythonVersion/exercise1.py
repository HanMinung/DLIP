from module import *

img = cv.imread('Image/rice.png',0)

thVal = 127
maxVal = 255

# Structure Element for Morphology
cv.getStructuringElement(cv.MORPH_RECT,(5,5))
kernel = np.ones((5,5),np.uint8)


# Gaussian blur
gblur = cv.GaussianBlur(img,(5,5), 0)        

# Apply Thresholding
ret,thresh = cv.threshold(gblur, thVal, maxVal, cv.THRESH_BINARY)

dstMorph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

# Plot results
titles = ['Original Image', 'GaussianBlur', 'Threshold image with blurred', 'Morphology']
images = [img, gblur, thresh, dstMorph]

for i in range(4):
    
    plt.subplot(2, 2, i+1),plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    
plt.show()
