## MULTIPLE IMAGES PANORAMA STITCHING
A multiple images panorama stitcher created to stitch any number of images without any order to be pregiven.
## HOW TO RUN ON LOCAL SYSTEM

To run the code change the file_path to the filepath where images are present and install all requirements as mentioned 
in requirements.txt file then run main.py result image will be stored in the same folder where scripts are present by 
the name "final_res.jpg"

## CODE DETAILS

The main code is divided into two parts all the functions are contained in the functions.py file while main code is in
main.py

The flow of code is as follows -<br>
#1) Storing every images data (name,image,index,keypoint, descriptors)<br>
   Take all the images present in the folder and store their data i.e name,image,index,keypoint and descriptors in
   custom data structure called as image_data. Key points and descriptors are calculated using harris corner detection
   function that will have detect the corners of the image and return the corresponding keypoints and descriptors 
   computed by inbuild SIFT function of the corner point detected by harris corner detector.
   
#2) Creation of Inlier matrix -<br>
   Order is detected using inlier matrix that is calculated by feature matching of two images. Features are matched
   using FLANN based matching once raw matches are detected they are further checked using ratio based checking. After
   getting matches they are distinguished as inliers and outliers by RANSAC algorithm and our gethomography function is 
   used to get the homography matrix. Every image is matched with every other image, and their inliers are stored in 
   the inlier matrix at co-ordinate of (image i index,image j index) and (image j index,image i index)
   
#3) Detection of order -<br>
   The image with a maximum count of inliers is pushed first in the order then respectively other images are selected
   who have maximum count of inlier from the images selected in the order.
   
#4) Stitching - <br>
   A big canvas is created and first image is placed in centre of it now keypoint and descriptor of new first image are 
   calculated. Now next image from order is selected as second image their features are matched and homography matrix 
   is calculated. Using homography matrix second image is warped. Now first and second images are blended together using
   our blend image code. To remove extra black bars from the canvas added earlier we grab the contour which has maximum
   area. Now resultant image is taken as first image and next image in order is selected as second image and process
   is continued.
   
#5) Storing final image - <br>
    Final resultant stitched image is saved by the name "final_res.jpg" in the same file where script is present.

