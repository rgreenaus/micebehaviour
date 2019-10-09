import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


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
            if filepath.endswith(".mp4") and 'debug' not in filepath:
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
    csv_out = "filename,left,middle,right\n"  # string to store csv output
    root_dir = "/docs/Dropbox/Kleo sociability test data/"
    out_dir = "/docs/Dropbox/Kleo sociability test data/"
    filenames = get_filenames(root_dir, out_dir, csv_out=csv_out)

    # load reference frame
    ref_frame = cv2.imread(root_dir + 'reference.png')[:, :, 0]
    mask_threshold = 50
    _, ref_frame = cv2.threshold(ref_frame, mask_threshold, 255, cv2.THRESH_BINARY)

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
        frame_width = 1068
        frame_height = 668
        fps = int(video_in.get(cv2.CAP_PROP_FPS))

        # read the pos of the top left and bottom right pixels
        with open(file_path[:-3] + 'txt', 'r') as pos_file:
            x_start, y_start = pos_file.readline().rstrip().split(',')
            x_start = int(x_start)
            y_start = int(y_start)
            x_end, y_end = pos_file.readline().rstrip().split(',')
            x_end = int(x_end)
            y_end = int(y_end)

        printnow(f'x={x_start},y={y_start} to x={x_end},y={y_end}')

        # OLD create the masks for the left, middle, and right portions
        # left_mask = np.full((frame_height, frame_width), False, dtype=np.bool)
        # centre_mask = np.full((frame_height, frame_width), False, dtype=np.bool)
        # right_mask = np.full((frame_height, frame_width), False, dtype=np.bool)
        # left_mask[76:548, 152:399] = True
        # centre_mask[76:548, 399:637] = True
        # right_mask[76:548, 637:887] = True

        # create the masks for the left, middle, and right portions
        mask = cv2.imread('mask.png')
        _, mask = cv2.threshold(mask[:, :, 0], 125, 255, cv2.THRESH_BINARY)
        _, components = cv2.connectedComponents(mask)
        left_mask = np.full((frame_height, frame_width), False, dtype=np.bool)
        centre_mask = np.full((frame_height, frame_width), False, dtype=np.bool)
        right_mask = np.full((frame_height, frame_width), False, dtype=np.bool)
        left_mask[components == 1] = True
        centre_mask[components == 2] = True
        right_mask[components == 3] = True

        # create the file handle to write out the debug video
        video_out = cv2.VideoWriter(file_path[:-4] + '_debug' + file_path[-4:], cv2.VideoWriter_fourcc(*'PIM1'), fps, (frame_width, frame_height), True)

        fc = 0
        counter = {'left': 0, 'centre': 0, 'right': 0}
        while True:
            # read the frame
            ret, raw_frame = video_in.read()
            if not ret:  # if we have hit the end
                break  # break the loop

            # convert each frame to greyscale
            frame = np.dot(raw_frame[..., :3], [0.299, 0.587, 0.114])
            # take only the video part of the frame
            frame = frame[y_start:y_end, x_start:x_end]
            # scale frame to the standard reference size
            frame = cv2.resize(frame, ref_frame.shape[::-1])

            # convert to unsigned int
            frame = frame.astype(np.uint8)
            _, frame = cv2.threshold(frame, mask_threshold, 255, cv2.THRESH_BINARY)

            # compare to reference
            diff_frame = np.abs(frame.astype(np.int16) - ref_frame.astype(np.int16))

            # determine the frame with the highest difference
            left_diff = np.sum(diff_frame[left_mask])
            centre_diff = np.sum(diff_frame[centre_mask])
            right_diff = np.sum(diff_frame[right_mask])

            out_frame = raw_frame[y_start:y_end, x_start:x_end]
            out_frame = cv2.resize(out_frame, ref_frame.shape[::-1])

            if left_diff > centre_diff and left_diff > right_diff:  # if left is the highest
                counter['left'] += 1
                out_frame[:, :, 0][left_mask] += 50
            elif centre_diff > right_diff:  # if middle is the highest
                counter['centre'] += 1
                out_frame[:, :, 0][centre_mask] += 50
            else:  # if right is the highest
                counter['right'] += 1
                out_frame[:, :, 0][right_mask] += 50
            out_frame[:, :, 0][out_frame[:, :, 0] > 255] = 255

            # write out coloured raw frame
            video_out.write(out_frame)

            fc += 1

        # add the time spent in each third to the list of results
        csv_out += f'{file_name},{counter["left"]},{counter["centre"]},{counter["right"]}\n'
        # close video_files
        video_in.release()
        video_out.release()

    # write out the results to a csv
    with open(out_dir + 'results.csv', 'w') as out_csv:
        out_csv.write(csv_out)


if __name__ == '__main__':
    main()


