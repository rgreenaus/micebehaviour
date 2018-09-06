import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import imageio

# frame 944 of the first video has a nice shot of the mouse


# def load_video(filename):
#     """
#     Takes a filename of a video and returns a 3d numpy array of the data
#     :param filename: a string of the full filepath to the video
#     :return: a numpy array representation of the video
#     """
#
#
#
#     # cv2.namedWindow('frame 10')
#     # cv2.imshow('frame 10', buf[900])
#     #
#     # cv2.waitKey(0)
#
#     return buf

def get_portions(arr, x, y):
    """
    takes a 2d array and returns 4 subsections - top left, top right, bottom left, bottom right
    """

    x = int(arr.shape[0]/2)
    y = int(arr.shape[1]/2)  # c for centre

    return [arr[0:x, 0:y], arr[0:x, y:], arr[x:, 0:y], arr[x:, y:]]


def imview(arr):
    plt.imshow(arr)
    plt.show()


def printnow(str, end="\n"):
    print(str, end=end)
    sys.stdout.flush()

def get_diff(arr1, arr2):
    """
    returns the sum of the absolute differences between all values
    """

    return np.sum(np.abs(arr1 - arr2))


def main():

    # video filepaths
    root_dir = "videos/"
    filenames = ["CM 193F Day3 T6.mpg"] # "CM 192F Day3 T4.mpg",
    ref_frame_nums = [510]  # 834, # a frame from the video that all other frames can be compared to
    # todo move to 4 mask files for non-rectangular quadrants
    masks = ["mask.png", "mask.png"]  # filenames of the mask files
    section_names = ["TL", "TR", "BL", "BR"]
    threshold = 100

    # todo load a manual file marking the start+end of testing

    # loop through each video
    for file_num in range(len(filenames)):
        printnow("Processing " + filenames[file_num])

        # load the video file
        printnow("\tLoading and converting to greyscale... ", end='')
        cap = cv2.VideoCapture(root_dir + filenames[file_num])
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        video = np.empty((frame_count, frame_height, frame_width), np.dtype('uint8'))
        video_c = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

        fc = 0
        while True:
            ret, tmp = cap.read()
            if not ret:
                break
            # build the coloured video matrix
            video_c[fc] = tmp
            # convert each frame to greyscale
            video[fc] = np.dot(tmp[..., :3], [0.299, 0.587, 0.114])
            fc += 1

        cap.release()

        cy = int(frame_width/2)  # center x coord
        cx = int(frame_height/2)  # centre y coord

        # convert the uint8 data to True/False
        mask = np.array(imageio.imread(root_dir + masks[file_num])) == 255

        printnow("done.")

        # Do the processing
        printnow("\tDetermining difference regions... ", end='')
        # build the reference frame
        ref_frame = video[ref_frame_nums[file_num]]
        ref_frame[mask == False] = 0  # set all the surrounding to zero
        ref_frame[ref_frame > threshold] = 255  # set all pixels above the threshold to white
        ref_frame[ref_frame <= threshold] = 0  # set all pixels below the threshold to black
        # plt.imsave("debug/ref_frame.png", ref_frame) # save for reference
        ref_frame_portions = get_portions(ref_frame, cx, cy)  # divide into portions

        most_movement = {}  # is a dict to hold entries in the frame_num:most_moved_quadrant format

        # threshold the video
        video[video > threshold] = 255
        video[video <= threshold] = 0

        for f_n in range(frame_count):
            # compare each frame to the reference frame
            frame = video[f_n]
            frame[mask == False] = 0  # set all pixels outside the mask to zero
            # plt.imsave("debug/%04d.png" % f_n, frame)
            frame_portions = get_portions(frame, cx, cy)  # split the frame into the quadrants
            diff = [0, 0, 0, 0]  # to hold the difference values for each quadrant
            for i in range(4):  # loop through each quadrant
                diff[i] = get_diff(frame_portions[i], ref_frame_portions[i])  # record the difference value
            # determine an integer for the quadrant with the highest difference
            highest = 0
            high_val = diff[0]
            for i in range(1, 4):
                if diff[i] > diff[highest]:
                    highest = i
                    high_val = diff[i]

            most_movement[f_n] = [highest, high_val]
        printnow("done.")

        # todo refine the quadrant movement - when the mouse is behind the fan there is a baseline amount of movement where the most recent quadrant should stay flagged

        # reconstruct the video in colour with the highest movement area highlighted
        printnow("\tMaking debug video and saving... ", end='')
        for f_n in range(frame_count):
            # turn the quadrant red
            if most_movement[f_n][0] == 0:  # UL
                video_c[f_n, 0:cx, 0:cy, 0] += 50
            elif most_movement[f_n][0] == 1:  # UR
                video_c[f_n, 0:cx, cy:, 0] += 50
            elif most_movement[f_n][0] == 2: # BL
                video_c[f_n, cx:, 0:cy, 0] += 50
            elif most_movement[f_n][0] == 3: # BR
                video_c[f_n, cx:, cy:, 0] += 50

        # convert video_c back into a video
        imageio.mimwrite("done-" + filenames[file_num], video_c, format="FFMPEG", fps=fps)
        printnow("done.")

        # todo generate and print results for time spent in each quadrant

        #
        # # todo covert to black and white based on a threshold
        #
        # # disp("\tGenerating difference between frames... ", end='')
        # # video_diff = np.diff(video_grey, axis=0)
        # # disp("done.")
        #
        # # todo find the biggest black blob inside the mask
        #
        # pass
        #
        # # todo this stuff
        # # threshold 128 is good, maybe 124 is a little better
        # # find the largest blob of a certain value
        # # https://stackoverflow.com/questions/20110232/python-efficient-way-to-find-the-largest-area-of-a-specific-value-in-a-2d-nump
        # # then compute the centroid
        # threshold = 128
        # video_bw = np.zeros(video_grey.shape, dtype=np.bool)
        # video_bw[np.where(video_grey > threshold)] = True
        #
        # for i in range(100):
        #     video_bw = np.zeros((video_grey.shape[1], video_grey.shape[2]), dtype=np.bool)
        #     video_bw[np.where(video_grey[994] > (100 + i))] = True
        #     plt.imshow(video_bw)
        #     plt.title(str(i))
        #     plt.show()
        #
        # # todo maybe difference isn't needed - just black&white it then look for the biggest black blob outside the masked area


if __name__ == '__main__':
    main()


