# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:29:26 2017
My first attempt at coding Hough Lines
@author: tsrivas
"""
import cv2
import numpy as np
import imutils


def hough_lines_acc(image, theta = np.arange(0, 170), rhoRes = 1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)
    cv2.imshow("Edged Image", edged), cv2.waitKey(0), cv2.destroyAllWindows()
    diag_length = np.int(np.ceil((np.sqrt((image.shape[0]-1)**2 + (image.shape[1]-1)**2))))
    print diag_length
    #diag_length = np.int(rhoRes*np.ceil(diag_length/rhoRes))
    rhos = np.arange(-diag_length, diag_length + rhoRes, rhoRes)
    height = len(rhos)
    print height
    rho = []
    thetacount = []
    count = 0
    H = np.zeros( (height, len(theta)), dtype = "uint8")
    for y in range(edged.shape[0]):
        for x in range(edged.shape[1]):
            if edged[y, x] == 255:
                for k,a in enumerate(theta):
                    #print k
                    arad = (np.pi / 180) * a # convert to radians
                    rhocurrent = np.int(x*np.cos(arad) + y*np.sin(arad))
                    
                    if rhocurrent == -241:
                        if a == 179:
                            print y, x
                            count = count+1
                        
                    
                    
                    #print rhocurrent
                    rho.append(rhocurrent)
                    
                    
                    H[(rhocurrent+ diag_length), k] +=1
    print count
    #print thetacount
    rho = set(rho)
    rho = list(rho)
    return H, rhos, theta

def show_hough_line(img, accumulator, peaks = None, rhos = None):
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(1, 2, figsize=(10, 10))

  ax[0].imshow(img, cmap=plt.cm.gray)
  ax[0].set_title('Input image')
  ax[0].axis('image')

  ax[1].imshow(
    accumulator, cmap='jet',
    extent=[(theta[-1]), (theta[0]), rhos[-1], rhos[0]])
  if len(peaks)>0:
      for j in range(len(peaks)):
          xcoord = peaks[j][1]
          ycoord = rhos[peaks[j][0]]
          ax[1].scatter(xcoord, ycoord, s = 30, c = u'r', marker = u'x' )
      print "Displaying first %d peaks" %(len(peaks))
      
          
  ax[1].set_aspect('equal', adjustable='box')
  ax[1].set_title('Hough transform')
  ax[1].set_xlabel('Angles (degrees)')
  ax[1].set_ylabel('Distance (pixels)')
  ax[1].axis('image')

  #plt.axis('off')
  #plt.savefig('imgs/output.png', bbox_inches='tight')
  plt.show()

# plt.imshow(Hscale, cmap = 'gray', extent = [90, -90, 478, -478]),plt.show()
def hough_peaks(accumulator, no_of_lines):
    peaks = []
    temp_max = 0
    temp_coord = []
    for i in range(no_of_lines):
        for x in range(H.shape[0]):
            for y in range(H.shape[1]):
                if H[x, y] > temp_max:
                    t = [x, y]
                    if t not in peaks:
                        temp_max = H[x, y]
                        temp_coord = t
        peaks.append(temp_coord)
        temp_coord = []
        temp_max = 0
    return peaks
                
                
#    for i in range(no_of_lines):
#        rhopeak, thetapeak = np.where(accumulator == max(accumulator.flatten()))
#        print accumulator[rhopeak[0], thetapeak[0]]
#        peaks[i] = rhopeak[0], thetapeak[0]
#        accumulator[rhopeak[0], thetapeak[0]] = 0
    
    return peaks
        
if __name__ == "__main__":
    image = cv2.imread('C:/Users/tsrivas/Documents/Thayjes/UdacityCV/ud810-master/course_images/ps1-input0.png')
    print image.shape    
    [H, rhos2, theta] = hough_lines_acc(image)
    #Hscale = H*255/H.max()
    peaks = hough_peaks(H.copy(), 8)
    #plt.imshow(Hscale, cmap = 'gray', extent = [theta[-1], theta[0], rhos[-1], rhos[0]]),plt.show()
    show_hough_line(image, H, peaks, rhos2)
    
        
    
        
    
