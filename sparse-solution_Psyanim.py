import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import argparse
import os

parser = argparse.ArgumentParser(description='Parameters ')
parser.add_argument('--video_path', default='/data/peng/SocialPerceptionModelingData/data/video_data_psyanim_socialscore/', type=str,
                    help='the folder includes videos ')
parser.add_argument('--result_root_path', default='/data/peng/SocialPerceptionModelingData/data/opticalflow/', type=str,
                    help='the folder includes extracted frames from videos ')






def get_optical_flow_images(video_path, result_path):

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

def convert_image2video(image_folder, video_name):
    # use ffmpeg to combine frame images into videos and save it into the same folder
    # Define command
    command = 'ffmpeg -y -r 30 -f image2 -s 600x450 -i \"{}\" -vcodec libx264 -crf 25 -pix_fmt yuv420p \"{}\" '.format(
        os.path.join(image_folder, "%d.png"), video_name)
    # Execute command
    subprocess.call(command, shell=True)


if __name__ == '__main__':
    #get the video path
    args = parser.parse_args()
    video_path_orignal = args.video_path
    result_root_path = args.result_root_path
    #get all videos in the video path
    video_list = os.listdir(video_path_orignal)
    for video in video_list:
        video_name = video.split(".")[0]
        #get the video path
        video_path = os.path.join(video_path_orignal, video)
        #get the result path
        result_path = os.path.join(result_root_path, video_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        #get the optical flow images
        get_optical_flow_images(video_path, result_path)
        #convert the optical flow images into video
        convert_image2video(result_path, os.path.join(result_root_path,video_name + '.mp4'))


