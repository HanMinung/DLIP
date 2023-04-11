# Load image
from module import *

img = cv.imread('../Image/coins.png',cv.COLOR_BGR2GRAY)

# Apply Canndy Edge
edges = cv.Canny(img,50,200)

# Plot Results
#cv.imshow('Edges',edges)
titles = ['Original','Edges']
images = [img, edges]

for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()