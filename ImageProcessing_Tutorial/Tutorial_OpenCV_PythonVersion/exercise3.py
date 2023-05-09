from module import *

src = cv.imread('Image/coin.jpg', cv.IMREAD_ANYCOLOR)

thVal = 190
maxVal = 255

# 500won, 100won, 50won, 10won sequence
coin500 = 0
coin100 = 0
coin50 = 0
coin10 = 0

srcGray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

cv.getStructuringElement(cv.MORPH_RECT,(5,5))
kernel = np.ones((5,5),np.uint8)

# Gaussian blur
gblur = cv.medianBlur(srcGray, 5)   

srcEqual = cv.equalizeHist(gblur)     

# Apply Thresholding
ret, thresh = cv.threshold(srcEqual, thVal, maxVal, cv.THRESH_BINARY)

dstMorph = cv.dilate(thresh, kernel, iterations = 1)
# dstMorph = cv.erode(dstMorph, kernel, iterations = 2)

# contour
contours, hierarchy = cv.findContours(dstMorph, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    
    length = cv.arcLength(cnt, True)
    
    if length >= 440 and length <= 500 :         
        color = (0, 0, 255)             # red
        coin500 = coin500 + 1
        
    elif length > 410 and length < 440 :     
        color = (0, 255, 0)             # green
        coin100 = coin100 + 1
        
    elif length >= 350 and length <= 400 :    
        color = (255, 0, 0)             # blue
        coin50 = coin50 + 1
        
    elif length >= 300 and length <= 350 :    
        color = (0, 255, 255)           # yellow
        coin10 = coin10 + 1
    
    if(length > 300)   :  cv.drawContours(src, [cnt], -1, color, 2)

money = 500 * coin500 + 100 * coin100 + 50 * coin50 + 10 * coin10
cv.putText(src, "Total money : " + str(money) + "[won]", [10, 50], cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
# print("Total money is : ", money, "[won] !")

cv.imshow('Contours', src)
cv.waitKey(0)
cv.destroyAllWindows()


# Plot results
# titles = ['Original Image', 'Histogram equalization','GaussianBlur', 'Threshold imgage', 'Morphology iamge']
# images = [src, srcEqual, gblur, thresh, dstMorph]

# for i in range(5) :
    
#     plt.subplot(3, 2, i+1),plt.imshow(images[i], 'gray', vmin=0, vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
    
# plt.show()



