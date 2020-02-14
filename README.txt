Code written and compiled on pyCharm 2019.1.3 running on macOS Mojave 10.14.6

Command to run the program
- python3 main.py

For the code
The program reads the images as coloured and grayscale image. It is using SIFT pyramid to make the points scale invariant and 
then using harris detector, it will calculate corners around the image using determinant and trace of the Harris matrix. The 
corners will cover more than 1 pixel, and therefore it will go under a non-maximum suppression to represent each corner as a 
single pixel. On the result obtained from non-max. suppression, it will again undergo adaptive non-max. suppression which 
takes euclidean distance between the points and if it's inserted in the list according to it's closeness to 0.

After obtaining all the feature points, it will construct a SIFT descriptor to compare the points in both the images. The 
orientation and magnitude are calculated in 16x16 window and the window is shifted according to the orientation of the feature 
point. Histogram bins of size 8 are used to calculate orientation. With descriptors in hand, it will compute the matches and 
draw matches using openCV's in built function.

For improved matches, it will compute the distances between the feature points that are selected as matches and compares if 
they have distance greater than a particular threshold and they are not optimal matches with sub-optimal ssd ratio. On basis of 
these two factors, it will decide if that feature point is needed.
