# Panorama using SIFT
In the script Panorama.py is a code that takes two images and stiches them into one panoramic image
We used in this script the SIFT algorithm and Homograph matrix in order to reach the result

1- first we take the images and resize them so they can be in the same height (big images are resized to 70% of the original size)

2- we turn them into gray scale in order for SIFT algorithm to work

3- we detect keypoints in each image 

4- we use knn to match between the similar regions in both images

5- after the knn finshes we will have for each right image region two most close regions in the left image

after that we take the best of both by using  distance1<ratio*distance2

6- we take the key points of the best regions

7- we create a homography matrix and use a wraping function with right image

8- we add the left image and stech it to the final result

9- saving the panorama in the given path

In order to run the script enter a right image path , left image path and the path of where you want to save the panorama 
example: Panorama.py 1/left.jpg 1/right 1/panora.jpg 
