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

First, I explore each RGB and HLS color channel to see whether which channel can generate distinct clear lane line. Among these channels, S channel can provide the best raw lane line detection. Then, I narrow down the potential color channels that will be used in this pipeline, including S,L,R,G,gray channels.

![alt text][image3]

To deep dive into S and L channel, I apply the color threshold and the output binary results in S (threshold=(100,255)) and L (threshold=(100,255)) channel are shown here:

![alt text][image4]

Likewise, applying the threshold of (170,255) in both R and G channel gives the binary image results like this:

![alt text][image5]

For gray channel, let's apply the sobel operator using x-orient magnitude, absolute magnitude, and direction. The result is shown below. The binary absolute sobel give the best result in this section.

![alt text][image6]

To combine all color and gradient technique in this section, I create the "process_color_and_gradient" function. After several experiment of finding the right combination between binary threshold and binary sobel result, I use S&L binary and Gray binary absolute, described by "Combined_binary[(S_binary & L_binary) | (Gray_binary_abs == 1)] = 1". The results by applying this function is shown here:

![alt text][image7]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, described in Step4: Perspective Transformation section.  The `warper()` function takes as inputs an image (`img`). The source (`src`) and destination (`dst`) points are choose manually by the following manner:


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 233, 713      | 300, 720      | 
| 560, 478      | 300, 1        |
| 729, 478      | 900, 1        |
| 1080, 718     | 900, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image8]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find a lane line, I use the warped image. Then I take a histogram of the bottom half of the image and find the histogram peak of the left and right halves of the histogram. Then, 9 sliding windows with margin of 100 and min number of pixel of 50 are applied to identify nonzero x-y pixels used for 2nd order polynomial fitting to determine the curve line. The result is shown below. To include all these step in this section, I create the "lane_line_process" function.

![alt text][image10]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The "cal_radius_offset" function is introducted here. Using the 2nd order polynomial fit (y-fit = Ay^2 + By + C), the radius of curvature is calculated by R = [1 + (2Ay+B)^2]^1.5/|2*A|

The position of the vehicle w.r.t the center is defined as offset which is determined by the difference between image center and the average position of left and right lane.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In conclusion, I put all steps together and define "lane_detection_pipeline".  Here is an example of my result on a test image:

![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video 

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
