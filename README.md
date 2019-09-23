# UCF ComputerVision Assignment1
This repo contains two python scripts: \
 - image_transform.py transforms a given image into a new image after rotation, scaling or shifing; \
 - mmse_and_gradient_des.py script implements Least Square method and Iterative gradient descent method to find a solution of a homogeneous linear system \
 The file data.txt has a 20 x 5 matrix X, and a 20 x 1 vector b. The output of the script is a vector h, that minimizes the norm
 of the error e=Xh-b.


The image_transform.py script makes use of the given cat image, image1.png and produces three output images:
output1.png (rotation by negative 45 degree), \
output2.png (scaling by a factor of 2), and \
output3.png (translation by 10 pixels in both positive x and positive y direction).


Instructions on how to run program
First, the following dependencies need to be installed to be able to run the program:
``` 
$ sudo apt-get install python3-pip
$ pip3 install numpy
$ pip3 install matplotlib
```
Then run the script
```
$ python3 image_transform.py
$ python3 mmse_and_gradient_des.py
```
