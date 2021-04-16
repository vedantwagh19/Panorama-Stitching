import os
import cv2
import functions
import  numpy as np


class image_data:
  def __init__(self,index,image,kp,desc,name):
    self.name=name
    self.index=index
    self.image=image
    self.kp=kp
    self.desc=desc

images_data=[]
index=0
file_path="test_multiple"   #file path of folders where images are present

print("Started Processing......")
print("Images to be stitched - ")
for i in os.listdir(file_path): #iterate through all images and store keypoints and desc
  cur_image=cv2.imread(file_path+"/"+i)
  print("("+str(index)+") : "+i)
  kp,desc=functions.harris_corner_detection(cur_image,3,0.04,1.5)
  images_data.append(image_data(index,cur_image,kp,desc,i))
  index+=1


total_images=int(index)
inlier_matrix=np.zeros((total_images,total_images))   #get inlier matrix
for i in range(0,total_images):
  for j in range(i+1,total_images):
    matches=functions.match_features(images_data[i].desc,images_data[j].desc)
    (H,total_inlier,status)=functions.get_homography(images_data[i].kp,images_data[j].kp,matches)
    inlier_matrix[i][j]=total_inlier
    inlier_matrix[j][i]=total_inlier

order=functions.find_order(inlier_matrix,total_images)  #find the order of stitching

print("Done computing order - ",end="")
print(order)

first_image=images_data[int(order[0])].image
first_image=functions.add_borders(first_image)        #place first image on big canvas
first_image_kp,first_image_desc=functions.harris_corner_detection(first_image,3,0.04,1.5) #get respective images kp and desciptor

for i in range(1,len(order)):
  second_image_index=int(order[i])  #select next image in order
  #print(second_image_index)
  second_image=images_data[second_image_index].image
  second_image_kp=images_data[second_image_index].kp
  second_image_desc=images_data[second_image_index].desc
  matches=functions.match_features(second_image_desc,first_image_desc)
  (H,inlier,status)=functions.get_homography(second_image_kp,first_image_kp,matches)
  res_image=functions.stitch(second_image,first_image,H)  #get resultant image
  if(i==total_images-1):
      first_image=res_image
      break
  else:
      #cv2.namedWindow("",cv2.WINDOW_NORMAL)
      #cv2.imshow("",res_image)
      #if cv2.waitKey(0) & 0xff == 27:
        #cv2.destroyAllWindows()
      first_image=functions.add_borders(res_image)  #consider resultant image as first image and place it on big canvas
      first_image_kp,first_image_desc=functions.harris_corner_detection(first_image,3,0.04,1.5)

cv2.imwrite("final_res.jpg",first_image)  #save the final image
cv2.namedWindow("",cv2.WINDOW_NORMAL)
cv2.imshow("",first_image)                #show the final image
if cv2.waitKey(0) & 0xff == 27:
  cv2.destroyAllWindows()

print("Done stitching Images")


