import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import imageio
import os

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


def printnow(s, end="\n"):
    print(s, end=end)
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
            if filepath.endswith(".mpg") and os.path.isfile(filepath[:-4] + ".txt"):
                csv_file_path = os.path.join(out_dir, subdir[len(root_dir):], file[:-4] + ".csv")
                if csv_out and os.path.isfile(csv_file_path):  # if we don't want to overwrite existing csv files
                    with open(csv_file_path, "r") as csv_in:
                        csv_out += csv_in.readline()
                else:
                    # get the start and stop times from the text file
                    # add this filepath to the return list along with the start and stop time read from its associated text file
                    with open(filepath[:-4] + ".txt", "r") as text_file_h:
                        start_t, end_t = text_file_h.readline().split(',')
                        rtn_list.append((filepath, int(start_t), int(end_t)))

    return rtn_list


def main():

    # video filepaths
    csv_out = "filename,top_secs,right_secs,bottom_secs,left_secs,start_time,end_time,duration\n"  # string to store csv output
    root_dir = "/docs/Dropbox/Small cues probe 1 for software analysis/" #"/docs/Dropbox/210, 212, 213 and txt files/Probe 2/"
    out_dir = "/docs/Dropbox/Small cues probe 1 for software analysis/" #"/docs/Dropbox/210, 212, 213 and txt files/Probe 2/"
    mask_dir = "masks/"
    filenames = get_filenames(root_dir, out_dir, csv_out=csv_out)
    ref_frame_nums = [510]  # 834, # a frame from the video that all other frames can be compared to
    threshold = 100
    mask_fn = "mask.png"  # filenames of the mask files
    # 4 mask files for non-rectangular quadrants
    mask_names = ["top", "right", "bottom", "left"]
    section_mask_fns = ["mask top.png", "mask right.png", "mask bottom.png", "mask left.png"]
    masks = {}
    for fn in range(len(section_mask_fns)):
        masks[mask_names[fn]] = np.array(imageio.imread(mask_dir + section_mask_fns[fn]))

    # loop through each video
    count = 1
    for file_path, start_t, end_t in filenames:
        file_name = file_path[len(root_dir):]
        printnow(f"Processing Video ({count}/{len(filenames)}) {file_name}")
        count += 1

        # load the video file
        printnow("\tLoading and converting to greyscale... ", end='')
        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # get the starting and ending frame numbers
        start_f = start_t * fps
        end_f = end_t * fps
        frame_count = end_f - start_f

        # only load the frames from start frame to end frame
        video = np.empty((frame_count, frame_height, frame_width), np.dtype('uint8'))
        video_c = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

        fc = 0
        while True:
            ret, tmp = cap.read()
            if not ret:
                break
            if fc < (start_f-1):  # if we haven't hit the starting frame yet
                fc += 1
                continue  # move to the next frame
            elif fc >= end_f:
                break  # we've reached the end of the window
            else:
                # build the coloured video matrix
                video_c[fc - start_f] = tmp
                # convert each frame to greyscale
                video[fc - start_f] = np.dot(tmp[..., :3], [0.299, 0.587, 0.114])
                fc += 1

        cap.release()

        printnow("done.")

        printnow("\tProcessing video... ", end='')
        quad_results = {}  # is a dict to hold entries in the frame_num:most_moved_quadrant format

        # threshold the video
        video[video > threshold] = 255
        video[video <= threshold] = 0

        debug_arr = np.zeros((frame_count, 4))  # stores all diff results in frame#, quad format

        for f_n in range(frame_count):  # for each frame
            highest = 0
            high_quad = ''
            debug_count = 0
            for name, mask in masks.items():  # for each mask
                # compare the current frame to the current mask
                frame = np.array(video[f_n])
                frame[mask == 0] = 0  # set all pixels outside the mask to zero
                d = get_diff(frame, mask)

                debug_arr[f_n, debug_count] = d
                debug_count += 1

                if d > highest:
                    highest = d
                    high_quad = name

            # save the high value and the
            quad_results[f_n] = [highest, high_quad]

        # using the debug data we'll remove the min from each quadrant to put them on a level footing
        debug_arr[:, 0] = debug_arr[:, 0] - np.min(debug_arr[:, 0])
        debug_arr[:, 1] = debug_arr[:, 1] - np.min(debug_arr[:, 1])
        debug_arr[:, 2] = debug_arr[:, 2] - np.min(debug_arr[:, 2])
        debug_arr[:, 3] = debug_arr[:, 3] - np.min(debug_arr[:, 3])
        for f_n in range(0, end_f - start_f):  # for each frame
            quad_results[f_n] = [np.max(debug_arr[f_n]), mask_names[np.argmax(debug_arr[f_n])]]
            
        printnow("done.")

        # todo refine the quadrant movement - when the mouse is behind the fan there is a baseline amount of movement where the most recent quadrant should stay flagged

        # reconstruct the video in colour with the highest movement area highlighted
        printnow("\tMaking debug video and saving... ", end='')
        quad_counter = [0, 0, 0, 0]  # number of frames spent in each quadrant
        red = 2  # the index of the red colour
        for f_n in range(frame_count):
            # todo convert the matrix to uint16, add 50, then convert back to uint8 (all >255 values should become 255)
            # turn the quadrant red
            key = 0
            if quad_results[f_n][1] == mask_names[0]:  # top
                key = 0
            elif quad_results[f_n][1] == mask_names[1]:  # right
                key = 1
            elif quad_results[f_n][1] == mask_names[2]:  # bottom
                key = 2
            elif quad_results[f_n][1] == mask_names[3]:  # left
                key = 3

            vid_cpy = np.array(video_c[f_n]).astype(dtype=np.uint16)  # get a colour copy of the current frame
            vid_cpy[masks[mask_names[key]] == 255, red] += 50  # add 50 to the intensity of red
            vid_cpy[vid_cpy[:, :, red] > 255, red] = 255  # reduce any red values > 255 to 255
            video_c[f_n] = vid_cpy.astype(dtype=np.uint8)  # copy back into video array

            quad_counter[key] += 1  # increment quadrant counter

        # convert video_c back into a video
        out_file_path = out_dir + file_name[:-4] + "_debug" + file_path[-4:]
        out_file_dirs = out_file_path[:out_file_path.rfind('/')]
        if not os.path.exists(out_file_dirs):
            os.makedirs(out_file_dirs)
        writer = cv2.VideoWriter(out_file_path, cv2.VideoWriter_fourcc(*'PIM1'), fps, (frame_width, frame_height), True)
        for v_f in range(video_c.shape[0]):  # for each frame in video_c
            writer.write(video_c[v_f])
        writer.release()
        printnow("done.")

        printnow(f"start time: {start_t}s end time: {end_t}s")
        for n_i in range(4):
            print("%s=%d" % (mask_names[n_i], quad_counter[n_i]), end=" ")
        print()
        for n_i in range(4):
            print("%s=%.2f%%" % (mask_names[n_i], (quad_counter[n_i]/len(quad_results))*100), end=" ")
        print()
        for n_i in range(4):
            print("%s=%.2fs" % (mask_names[n_i], quad_counter[n_i]/fps), end=" ")
        print('\n')

        csv_line = "%s,%.2f,%.2f,%.2f,%.2f,%d,%d,%d\n" % (file_name[file_name.rfind('/')+1:-4],
                                                          quad_counter[0] / fps,
                                                          quad_counter[1] / fps,
                                                          quad_counter[2] / fps,
                                                          quad_counter[3] / fps,
                                                          start_t,
                                                          end_t,
                                                          end_t - start_t)
        csv_out += csv_line
        csv_line_file = open(out_dir + file_name[:-3] + "csv", "w")
        csv_line_file.write(csv_line)
        csv_line_file.close()

        del video_c
        del video

    # save csv out
    csv_file = open(out_dir + "output.csv", "w")
    csv_file.write(csv_out)
    csv_file.close()
    printnow("\n\n" + csv_out)


if __name__ == '__main__':
    main()


