
import cv2 as cv
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import subprocess
import argparse
import os
import pandas as pd
parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--video_path', default='/data/peng/SocialPerceptionModelingData/data/video_data_psyanim_socialscore/', type=str,
                    help='the folder includes videos ')
parser.add_argument('--result_root_path', default='/data/peng/SocialPerceptionModelingData/data/opticalflow_dense/csv/', type=str,
                    help='the folder includes extracted frames from videos ')






def get_sparse_optical_flow_images(video_path, result_path):

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 3)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize = (3,3), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # The video feed is read in as a VideoCapture object
    cap = cv.VideoCapture(video_path)
    # Variable for color to draw optical flow track
    color = (0, 255, 0)
    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, first_frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    # Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
    # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
    prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    # Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
    mask = np.zeros_like(first_frame)

    frame_count = 0
    while(cap.isOpened()):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        frame_count = frame_count + 1
        if frame is None:
            break
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Calculates sparse optical flow by Lucas-Kanade method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
        prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
        next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        # Selects good feature points for previous position
        good_old = prev[status == 1].astype(int)
        # Selects good feature points for next position
        good_new = next[status == 1].astype(int)
        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            mask = cv.line(mask, (a, b), (c, d), color, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            frame = cv.circle(frame, (a, b), 3, color, -1)
        # Overlays the optical flow tracks on the original frame
        output = cv.add(frame, mask)
        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)
        # Opens a new window and displays the output frame
        # cv.imshow("sparse optical flow", output)

        outimage = os.path.join(result_path, str(frame_count) + ".png")
        cv.imwrite(outimage, output)

        # plt.imshow(output, interpolation='bicubic')
        # plt.savefig(outimage)
        # plt.show()

        # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
        # if cv.waitKey(33) & 0xFF == ord('q'):
        #     break
    #
    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()


def find_unique_pixel_grayscale(data):
    # Convert the image into a numpy array
    data = np.array(data)

    # Get the unique pixel values
    unique_pixels = np.unique(data)

    # Print the unique pixel values
    for pixel in unique_pixels:
        print(pixel)

def find_unique_pixel_RGB(data):


    # Get the unique pixel values
    unique_pixels = np.unique(data.reshape(-1, data.shape[2]), axis=0)

    # Print the unique pixel values
    for pixel in unique_pixels:
        print(pixel)

def find_black_grey_pixel(data):


    # Define the color ranges for black and grey
    black = [0, 0, 0]
    grey = [128, 128, 128]

    # Create an empty list to hold the coordinates of the black and grey pixels
    black_pixels = []
    grey_pixels = []

    # Loop through each pixel in the image
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # If the pixel is black or grey, add its coordinates to the respective list
            if (data[i, j] == black).all():
                black_pixels.append((i, j))
            elif (data[i, j] == grey).all():
                grey_pixels.append((i, j))

    # Now you have a list of all the black and grey pixels in the image
    print('Black pixels:', black_pixels)
    print('Grey pixels:', grey_pixels)


def get_dense_optical_flow_images(video_path, result_path):
    # The video feed is read in as a VideoCapture object
    cap = cv.VideoCapture(video_path)
    # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
    ret, first_frame = cap.read()
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(first_frame)
    arrow_mask = np.zeros_like(first_frame)  # new mask for arrows
    # Sets image saturation to maximum
    mask[..., 1] = 255
    frame_count = 0
    # Define a list to store data from each frame
    data_list = []
    while (cap.isOpened()):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()
        frame_count = frame_count + 1
        if frame is None:
            break

        # Opens a new window and displays the input frame
        # cv.imshow("input", frame)
        # Converts each frame to grayscale - we previously only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Find gray and black pixels
        # find_unique_pixel_grayscale(gray)
        # plt.imshow(gray)
        # plt.show()


        # Calculates dense optical flow by Farneback method
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
        #So what you are actually getting is a matrix that has the same size as your input frame.
        #Each element in that flow matrix is a point that represents the displacement of that
        # pixel from the prev frame. Meaning that you get a point with x and y values
        # (in pixel units) that gives you the delta x and delta y from the last frame.
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Find gray and black pixels
        gray_indices = np.where((gray < 160) & (gray >= 140))
        black_indices = np.where(gray < 60)

        # Append the data for gray pixels
        for i in range(len(gray_indices[0])):
            y, x = gray_indices[0][i], gray_indices[1][i]
            dx, dy = flow[y, x]
            data_list.append([frame_count, x, y, 'gray', dx, dy])

        # Append the data for black pixels
        for i in range(len(black_indices[0])):
            y, x = black_indices[0][i], black_indices[1][i]
            dx, dy = flow[y, x]
            data_list.append([frame_count, x, y, 'black', dx, dy])


        # Visualization of optical flow as arrows
        # flow vectors every 10 pixels.
        step = 10
        y, x = np.mgrid[step / 2:frame.shape[0]:step, step / 2:frame.shape[1]:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        # create line endpoints
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)

        # create line mask
        arrow_mask = np.zeros_like(frame)
        for (x1, y1), (x2, y2) in lines:
            cv.arrowedLine(arrow_mask, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.5)

        # overlay line mask on frame
        output_arrow_overlay = cv.add(frame, arrow_mask)

        # Computes the x magnitude and y  angle directions of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
        # Sets image hue according to the optical flow y direction
        mask[..., 0] = angle * 180 / np.pi / 2
        # Sets image value according to the optical flow x direction (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
        # Opens a new window and displays the output frame
        # cv.imshow("dense optical flow", rgb)

        outimage = os.path.join(result_path, str(frame_count) + ".png")
        cv.imwrite(outimage, rgb)
        # Overlays the optical flow tracks on the original frame
        output_overlay = cv.add(frame, rgb)
        outimage_overlay_output_path = os.path.join(result_path, "overlay_" + str(frame_count) + ".png")
        cv.imwrite(outimage_overlay_output_path, output_overlay)


        # Save arrow image
        arrow_outimage = os.path.join(result_path, "arrow_" + str(frame_count) + ".png")
        cv.imwrite(arrow_outimage, arrow_mask)

        arrow_outimage_overlay_output_path = os.path.join(result_path, "overlay_arrow_" + str(frame_count) + ".png")
        cv.imwrite(arrow_outimage_overlay_output_path, output_arrow_overlay)

        # Updates previous frame
        prev_gray = gray



    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()
    # Save the data into a single CSV file
    df_flow = pd.DataFrame(data_list, columns=['frame', 'x', 'y', 'color', 'dx', 'dy'])

    return df_flow #return the dataframe of optical flow


def convert_image2video(image_folder, video_name):
    # use ffmpeg to combine frame images into videos and save it into the same folder
    # Define command
    command = 'ffmpeg -y -r 30 -f image2 -s 600x450 -i \"{}\" -vcodec libx264 -crf 25 -pix_fmt yuv420p \"{}\" '.format(
        os.path.join(image_folder, "%d.png"), video_name)
    # Execute command
    # subprocess.call(command, shell=True)
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.output)

    command = 'ffmpeg -y -r 30 -f image2 -s 600x450 -i \"{}\" -vcodec libx264 -crf 25 -pix_fmt yuv420p \"{}\" '.format(
        os.path.join(image_folder, "overlay_arrow_%d.png"), video_name[:-4]+'_overlay_arrow.mp4')
    # Execute command
    # subprocess.call(command, shell=True)
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.output)


    command = 'ffmpeg -y -r 30 -f image2 -s 600x450 -i \"{}\" -vcodec libx264 -crf 25 -pix_fmt yuv420p \"{}\" '.format(
        os.path.join(image_folder, "overlay_%d.png"), video_name[:-4]+'_overlay.mp4')
    # Execute command
    # subprocess.call(command, shell=True)
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.output)

if __name__ == '__main__':
    #get the video path
    args = parser.parse_args()
    video_path_orignal = args.video_path
    result_root_path = args.result_root_path
    #get all videos in the video path
    video_list = os.listdir(video_path_orignal)
    for video in video_list:
        video_name = video.split(".")[0]
        print(video_name)
        #get the video path
        video_path = os.path.join(video_path_orignal, video)
        #get the result path
        result_path = os.path.join(result_root_path, video_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        #get the optical flow images
        print("get the optical flow images")
        df_flow = get_dense_optical_flow_images(video_path, result_path)
        # After processing all frames, save the DataFrame to a CSV file
        df_flow.to_csv(os.path.join(result_root_path, video_name + "_optical_flow.csv"), index=False)
        #convert the optical flow images into video
        print("convert the optical flow images into video")
        convert_image2video(result_path, os.path.join(result_root_path,video_name + '.mp4'))



