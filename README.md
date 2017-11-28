# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]:  ./pic_writeup/pic1_chessboard_cal.png        "Camera Calibration"
[image2]:  ./pic_writeup/pic2_dis_vs_undist.png         "Distortion Correction"
[image3]:  ./pic_writeup/pic3_color_channels.png        "Explore Color Channel"
[image4]:  ./pic_writeup/pic4_S_L_threshold.png         "S-L Threshold"
[image5]:  ./pic_writeup/pic5_R_G_threshold.png         "R-G Threshold"
[image6]:  ./pic_writeup/pic6_gray_threshold.png        "Gray Sobel Threshold"
[image7]:  ./pic_writeup/pic7_color_and_threshold.png   "Color and Threshold Process"
[image8]:  ./pic_writeup/pic8_warped_img.png            "Warped Image"
[image9]:  ./pic_writeup/pic9_hist.png                  "Histogram X"
[image10]: ./pic_writeup/pic10_window.png               "Lane Line Curve Fitting (Window)"
[image11]: ./pic_writeup/pic11_window2.png              "Lane Line Curve Fitting (Filled-In Window)"
[image12]: ./pic_writeup/pic12_org_vs_final.png         "Original vs Final Image"

[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "Main_Advanced_Lane_Finding.ipynb".  

Note, the calibration chessboard image files include 9x6 corners, unlike the 8x6 corners used in the lesson. First, the object points and imgpoints are introduced as follows. The object points are the (x,y,z) coordinates of the chessboard corners in the world where the imgpoints are the (x,y) coordinateds of corresponding corner on the image plane, assuming z=0. I use the command "cv2.findChessboardCorners" to find all corners in the image plane. Then, append objp to objpoints, and corners to imgpoints. Finally for this step, I use "cv2.calibrateCamera" to calculate the camera calibration mapping. 

cv2.drawChessboardCorners will draw each corner points on the image plane like this:
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To correct the distortion, I use "cv2.undistort" command and the result is shown here:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

First, I explore each RGB and HLS color channel to see whether which channel can generate distinct clear lane line.
![alt text][image3]



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
