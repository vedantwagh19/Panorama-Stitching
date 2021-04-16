
import numpy as np
import imutils
from scipy import signal
import cv2
import math

class detections:            #custom data structure to store harris corners x,y co-ordinates and r value
    def __init__(self,x,y,r):
        self.x=x
        self.y=y
        self.r=r

class image_data:             #custom data structure to store image data
    def __init__(self,index,image,kp,desc):
        self.index=index
        self.image=image
        self.kp=kp
        self.desc=desc

def find_max(arr,vis):          #finding max of array and return index
    m=0
    index=0
    for i in range(0,len(arr)):
        if(m<arr[i] and vis[i]==0):
            m=arr[i]
            index=i
    return index

def sobel_x(img_grey):                                        #get x derivative using sobel
    filter_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    res_x=signal.convolve2d(img_grey,filter_x,mode='same')
    return res_x

def sobel_y(img_grey):                                         #get y derivative using sobel
    filter_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    res_y=signal.convolve2d(img_grey,filter_y,mode='same')
    return res_y

def prewitt_x(img_grey):                                      #get x derivative using prewitt
    filter_x=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    res_x=signal.convolve2d(img_grey,filter_x,mode='same')
    return res_x

def prewitt_y(img_grey):                                      #get y derivative using prewitt
    filter_y=np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    res_y=signal.convolve2d(img_grey,filter_y,mode='same')
    return res_y

def harris_corner_detection(image,window_size,k,sigma): #detecting harris corners and converting them to keypoints and descriptors

    img_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_grey_8bit = cv2.normalize(img_grey, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    img_grey=np.float64(img_grey)
    height,width=img_grey.shape

    I_x=sobel_x(img_grey)
    I_y=sobel_y(img_grey)

    I_xx=np.square(I_x)
    I_yy=np.square(I_y)
    I_xy=np.multiply(I_x,I_y)

    I_xx=cv2.GaussianBlur(I_xx,(9,9),sigma)
    I_yy=cv2.GaussianBlur(I_yy,(9,9),sigma)
    I_xy=cv2.GaussianBlur(I_xy,(9,9),sigma)

    offset=int(window_size/2)
    sift = cv2.SIFT_create()

    kp=[]
    corners=[]
    max_r=-math.inf

    for x in range(offset,height-offset):
        for y in range(offset,width-offset):
            W_xx=np.sum(I_xx[x-offset:x+1+offset, y-offset:y+1+offset])
            W_yy=np.sum(I_yy[x-offset:x+1+offset, y-offset:y+1+offset])
            W_xy=np.sum(I_xy[x-offset:x+1+offset, y-offset:y+1+offset])

            det=(W_xx*W_yy)-(W_xy**2)
            trace=W_xx+W_yy

            r=det-k*(trace**2)
            max_r=max(r,max_r)
            corners.append(detections(x,y,r))

    for i in corners:
        if(i.r > 0.01*max_r):
          kp.append(cv2.KeyPoint(i.y,i.x,10))

    kp, desc = sift.compute(img_grey_8bit,kp)

    #res=cv2.drawKeypoints(img_grey_8bit,kp,None)
    #cv2.imshow("",res)
    #if cv2.waitKey(0) & 0xff == 27:
      #cv2.destroyAllWindows()

    return kp,desc


def match_features(desc1,desc2): #matching feature from two images

    index_parameter = dict(algorithm = 1, trees = 5)  #using FLANN based matcher
    search_paramters = dict()
    flann = cv2.FlannBasedMatcher(index_parameter,search_paramters)
    raw_matches = flann.knnMatch(desc1,desc2,k=2)
    matches=[]

    for (m,n) in raw_matches:
      if m.distance < 0.7*n.distance:
        matches.append(m)
    return matches


def get_homography(kp1,kp2,matches):  #getting homography matrix and remove outliers using RANSAC

    kp1 = np.float64([kp.pt for kp in kp1])
    kp2 = np.float64([kp.pt for kp in kp2])
    #print(len(matches))
    Temp=[]
    if len(matches)>4:
       pt1 = np.float64([kp1[m.queryIdx] for m in matches])
       pt2 = np.float64([kp2[m.trainIdx] for m in matches])

       (H, status) = cv2.findHomography(pt1, pt2, cv2.RANSAC,2.0)
       total_inlier=0
       for i in range(0,status.shape[0]):
           if(status[i][0]):
               total_inlier+=1
       return (H,total_inlier,status)
    else:
        return (Temp,0,"None")


def remove_blackbars(image):   #grab the main stitched image by grabbing the main contour
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY)[1]

    contour = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)

    c = max(contour, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    image=image[y:y + h, x:x + w]
    #image=min_max_border(image,h,w)
    return image

def min_max_border(image,h,w):  #remove extra black border in top and bottom after extracting the main stitched image
    count1=0
    count2=0

    for i in range(0,h):
         if(image[i][0][1]==0 and image[i][0][2]==0 and image[i][0][2]==0):
            count1+=1
         else:
            break
         if(image[i][w-1][1]==0 and image[i][w-1][2]==0 and image[i][w-1][2]==0):
          count2+=1
         else:
           break
    start=max(count1,count2)

    count1=0
    count2=0
    for i in range(h-1,0,-1):
        if(image[i][0][1]==0 and image[i][0][2]==0 and image[i][0][2]==0):
            count1+=1
        else:
            break
        if(image[i][w-1][1]==0 and image[i][w-1][2]==0 and image[i][w-1][2]==0):
            count2+=1
        else:
            break
    end=max(count1,count2)

    image=image[start:h-end, 5:w-5]
    return image

def blend_image(prim_image,warp_image):    #blend two images i.e first image and the warped image
    temp_image=np.zeros(prim_image.shape, dtype=np.uint8)
    h=prim_image.shape[0]
    w=prim_image.shape[1]

    for i in range(0,h):
       for j in range(0,w):
           sum1=int(prim_image[i][j][0])+int(prim_image[i][j][1])+int(prim_image[i][j][2])
           sum2=int(warp_image[i][j][0])+int(warp_image[i][j][1])+int(warp_image[i][j][2])
           if(sum1>sum2):
                temp_image[i][j][:]=prim_image[i][j][:]
           else:
                temp_image[i][j][:]=warp_image[i][j][:]

    return temp_image


def stitch(image1,image2,H):   #stitch two images together i.e warp 1st image wrt to secong image and blend two images
    res_image=cv2.warpPerspective(image1,H,(image2.shape[1],image2.shape[0]))
    res_image=blend_image(image2,res_image)
    res_image=remove_blackbars(res_image)
    return res_image

def find_closest(vis,images_data,cur_desc):   #find the closest matching image

    max_matches=[]
    res_index=-1
    index=0
    for i in images_data:
        if(vis[index]<1):
            matches=match_features(i.desc,cur_desc)
            if len(max_matches)<len(matches):
                max_matches=matches
                res_index=index

        index+=1

    return (max_matches,res_index)

def add_borders(image):    #place the image on big canvas
    #print(image.shape)
    h=image.shape[0]
    w=image.shape[1]
    canvas=np.zeros((h*3,w*3,3),dtype='uint8')
    canvas[1*h:2*h,1*w:2*w]=image

    return canvas

def find_order(inlier_matrix,total_images):   #find the order of stitching image based on inlier matrix

    order=np.zeros(total_images)
    vis=np.zeros(total_images)
    inlier_sum=np.sum(inlier_matrix,axis=1)
    cur_index=find_max(inlier_sum,vis)

    order[0]=cur_index
    vis[cur_index]=1

    sec_inliers=[]
    for i in range(0,total_images):
        sec_inliers.append(inlier_matrix[cur_index][i])

    for i in range(1,total_images):
      cur_index=find_max(sec_inliers,vis)
      order[i]=cur_index
      vis[cur_index]=1

      sec_inliers+=inlier_matrix[cur_index][:]

    return order















