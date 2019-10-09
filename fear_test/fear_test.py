import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle


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


def get_filenames(root_dir, out_dir, csv_out=None):
    """
    returns the list of files to analyse
    :param root_dir: the path to the root directory of the folders that contains the videos
    :param out_dir: the directory the output is written to
    :param csv_out: if passed a string this function will not return files that already have csvs and will instead read them into the csv_out string
    :return: the list of file paths for analysis
    """

    rtn_list = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            filepath = os.path.join(subdir, file)
            # is it a mpg file with associated txt file?
            if filepath.endswith(".mov") and 'debug' not in filepath:
                # csv_file_path = os.path.join(out_dir, subdir[len(root_dir):], file[:-4] + ".csv")
                # if csv_out and os.path.isfile(csv_file_path):  # if we don't want to overwrite existing csv files
                #     with open(csv_file_path, "r") as csv_in:
                #         csv_out += csv_in.readline()
                # else:
                    # add this path to the rtn_list
                rtn_list.append(filepath)

    return rtn_list


def get_masks():
    """
    Returns a dictionary of each binary mask with the identifier as the key
    :return:
    """

    rtn_dict = {'left': cv2.imread('masks/left.png')[:, :, 0],
                'middle': cv2.imread('masks/middle.png')[:, :, 0],
                'right': cv2.imread('masks/right.png')[:, :, 0]}
    return rtn_dict


def main():

    # video filepaths
    # todo
    csv_out = "filename,left,middle,right\n"  # string to store csv output
    root_dir = "/docs/Dropbox/fear conditioning test vids/Fear Conditioning Test (Final)/"
    out_dir = "/docs/Dropbox/fear conditioning test vids/Fear Conditioning Test (Final)/"
    filenames = get_filenames(root_dir, out_dir, csv_out=csv_out)

    # loop through each video
    count = 1
    for file_path in filenames:
        file_name = file_path[len(root_dir):]
        printnow(f"Processing Video ({count}/{len(filenames)}) {file_name}")
        count += 1

        # load the video file
        printnow("\tLoading and converting to greyscale... ", end='')
        video_in = cv2.VideoCapture(file_path)
        frame_count = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_in.get(cv2.CAP_PROP_FPS))
        print(f'width={frame_width} height={frame_height} fps={fps}', end='')

        # load mask
        mask = cv2.imread('circle.png')[:, :, 0]  # take one of the three channels from the mask
        mask = mask > 0  # convert the mask into boolean values
        # set the mask threshold
        mask_threshold = 50

        # create the file handle to write out the debug video
        video_out = cv2.VideoWriter(file_path[:-4] + '_debug' + file_path[-4:], cv2.VideoWriter_fourcc(*'PIM1'), fps, (frame_width, frame_height), True)

        fc = 0
        centroid_list = []  # keeps a list of the centroids from the mouse blob
        prev_thresh = None  # stores the previous frame's thresholded image
        thresh_diff_list = []  # stores a list of the diffs between the thresholded frames
        prev_frame = None
        frame_diff_list = []
        while True:
            # read the frame
            ret, raw_frame = video_in.read()
            if not ret:  # if we have hit the end
                break  # break the loop

            # convert frame to greyscale
            frame = np.dot(raw_frame[..., :3], [0.299, 0.587, 0.114])
            # convert to unsigned int
            frame = frame.astype(np.uint8)
            # perform a binary threshold
            _, thresh = cv2.threshold(frame, mask_threshold, 255, cv2.THRESH_BINARY)
            # apply the mask
            thresh[~mask] = 255
            # invert the frame
            thresh = cv2.bitwise_not(thresh)

            # find the largest blob in the centre of the frame
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
            stats = stats[1:]  # drop the largest blob off (the background)
            mouse_idx = np.unravel_index(stats.argmax(), stats.shape)[0] + 1  # get the index of the mouse blob
            # save the position of the blob
            centroid_list.append(centroids[mouse_idx])

            # save the diff between this frame and the last to the frame_diff_list
            if prev_thresh is not None:
                thresh_diff_amount = np.sum(np.abs(prev_thresh[mask] - thresh[mask]))
                thresh_diff_list.append(thresh_diff_amount)
                if thresh_diff_amount < 40000:
                    raw_frame[:, :, 0][mask] += 50
                    raw_frame[:, :, 0][raw_frame[:, :, 0] > 255] = 255
                frame_diff_amount = np.sum(np.abs(prev_frame[mask] - frame[mask]))
                frame_diff_list.append(frame_diff_amount)
            prev_thresh = thresh
            prev_frame = frame

            # write out raw frame (coloured if no movement detected)
            video_out.write(raw_frame)

            fc += 1

        # todo
        pickle.dump({'centroid_list': centroid_list,
                     'frame_diff_list': frame_diff_list,
                     'thresh_diff_list': thresh_diff_list}, open(file_path[:-4] + '_debug.pickle', 'wb'), )
        # add the time spent in each third to the list of results
        # csv_out += f'{file_name},{counter["left"]},{counter["centre"]},{counter["right"]}\n'
        # close video_files
        video_in.release()
        video_out.release()

    # write out the results to a csv
    with open(out_dir + 'results.csv', 'w') as out_csv:
        out_csv.write(csv_out)


if __name__ == '__main__':
    main()
