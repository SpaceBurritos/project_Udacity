import cv2
import time
import numpy as np
import pandas as pd

#get the angle of those vectors
def Angle_converter(v1_x, v1_y, v2_x, v2_y):
    return np.arctan2(v2_y, v2_x) - \
           np.arctan2(v1_y, v1_x)

# With respect to the right_side
def Perception(right_side, left_side, center):
    proportion = abs((center - left_side) / (center - right_side))
    return proportion

#Gets if the pose-link is good (green) or bad (red)
def technique_color(technique_last_row, pair):
    body_parts = {
    "right_shoulder": [2,3], "right_elbow": [3,4], 'hands': [4,7],
    'left_shoulder': [5,6], 'left_elbow': [6,7],
    'right-side_hip': [8, 9], 'right_knee': [9,10], 'feets': [10,13],
    'left-side_hip': [11-12], 'left_knee': [12-13]
    }
    bad_color = (0,0,255)
    good_color = (0,255,0)
    line = good_color

    if pair == body_parts["right_shoulder"] or pair == body_parts["left_shoulder"]:
        if technique_last_row[0] == "Bad":
            line = bad_color
    elif pair == body_parts["right_elbow"] or pair == body_parts["left_elbow"]:
        if technique_last_row[1] == "Bad":
            line = bad_color
    elif pair == body_parts["hands"]:
        if technique_last_row[2] == "Bad":
            line = bad_color
    return line

def model(vid_source, write_bool=False):
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 14
    POSE_PAIRS = [[1, 0], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
              [12, 13]]
    body_parts_col = ['head_x', 'head_y', 'neck_x', 'neck_y', 'right_shoulder_x', 'right_shoulder_y', 'right_elbow_x',
                      'right_elbow_y', 'right_hand_x', 'right_hand_y', 'left_shoulder_x', 'left_shoulder_y',
                      'left_elbow_x',
                      'left_elbow_y', 'left_hand_x', 'left_hand_y', 'right-side_hip_x', 'right-side_hip_y',
                      'right_knee_x',
                      'right_knee_y', 'right_feet_x', 'right_feet_y', 'left-side_hip_x', 'left-side_hip_y',
                      'left_knee_x',
                      'left_knee_y', 'left_feet_x', 'left_feet_y']
    #Base dataframe
    df = pd.DataFrame(columns=body_parts_col)

    inWidth = 368
    inHeight = 368
    threshold = 0.1
    first_frame = True

    #Gets the stage in which the person is in the excercise
    counter = 0
    past_number = np.inf
    tol_hand_pos = 2 #Tolerance for the position of the hand
    tol_height_hands = 0.2 #Tolerance for the height of the hands
    tol_elbow_deg = 0.5 #Tolerance of the degree between the elbows
    stages = [['starting', 0], ['concentric', 1], ['isometric', 0], ['eccentric', -1], ['finalizing', 0]] #Stages of the excercise

    input_source = vid_source
    cap = cv2.VideoCapture(input_source)
    hasFrame, frame = cap.read()

    vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    while cv2.waitKey(1) < 0:
        t = time.time()
        hasFrame, frame = cap.read()
        frameCopy = np.copy(frame)
        if not hasFrame:
            cv2.waitKey()
            break
        if first_frame:
            first_frame = False
            continue

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                  (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]
        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold :
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append((None, None))

        df2 = pd.DataFrame(np.array(points).flatten()[np.newaxis, :], columns=body_parts_col)
        df = df.append(df2, ignore_index=True)

######################################################################################################################


        #if first_frames <= 0:
            # body_parts_speeds_col = [label[:-1] + 'd' + label[-1] for label in body_parts_col]



        #  def distance_hand_shoulder():

        df_smoothed = df.ewm(com=0.3,
                             min_periods=1).mean()  # This is created to alleviate the errors on the movement from
        # the model
        speeds = df.diff(axis=0,
                                  periods=1)  # Speed relative to the dt of the frames and the dx from the pixels
        lengths = df.copy()  # Will be used to know the length of the body parts

        speeds.loc[(speeds['left_hand_y'] > 0) & (speeds['right_hand_y'] > 0), 'direction'] = '-1'  # Up
        speeds.loc[(speeds['left_hand_y'] < 0) & (speeds['right_hand_y'] < 0), 'direction'] = '1'  # Down
        speeds.loc[(speeds['left_hand_y'] == 0) & (speeds['right_hand_y'] == 0), 'direction'] = '0'  # Is not moving
        speeds = speeds.fillna(method='ffill')

        last_row_speed = speeds.iloc[-1]

        if last_row_speed[0] != 0:
            if last_row_speed[-1] != past_number:
                past_number = last_row_speed[-1]
                stage = stages[counter][0]
                counter += 1
            speeds.at[-1, 'direction'] = stage  # The rest of the positions

        loc = 2
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            lengths.iloc[:, loc] = df_smoothed.iloc[:, partB * 2] - df_smoothed.iloc[:, partA * 2]
            lengths.iloc[:, loc + 1] = df_smoothed.iloc[:, partB * 2 + 1] - df_smoothed.iloc[:, partA * 2 + 1]
            loc += 2

        lengths["middle-hip_x"] = (lengths["right-side_hip_x"] + lengths["left-side_hip_x"]) / 2
        lengths["middle-hip_y"] = (lengths["right-side_hip_y"] + lengths["left-side_hip_y"]) / 2
        lengths.drop('head_x', 1, inplace=True)
        lengths.drop('head_y', 1, inplace=True)
        lengths.loc[np.abs(lengths['right_shoulder_x'] - lengths['right_hand_x'] -
                    lengths['left_shoulder_x'] - lengths['left_hand_x']) < tol_hand_pos, 'technique'] = "Good"
        # This angle will be used as base for the rest of the angles, this is the angle between the shoulders and the
        # middle of the hips
        angles = pd.DataFrame(np.arctan2(lengths["right_shoulder_y"], lengths["right_shoulder_x"]) -
                              np.arctan2(lengths["middle-hip_y"], lengths["middle-hip_x"]), columns=["perpendicular"])
        angles["right_shoulder"] = Angle_converter(lengths["right_shoulder_x"], lengths["right_shoulder_y"],
                                                   lengths["right_elbow_x"], lengths["right_elbow_y"])
        angles["left_shoulder"] = Angle_converter(lengths["left_elbow_x"], lengths["left_elbow_y"],
                                                  lengths["left_shoulder_x"], lengths["left_shoulder_y"])
        angles["right_arm"] = Angle_converter(lengths["right_elbow_x"], lengths["right_elbow_y"],
                                              lengths["right_hand_x"], lengths["right_hand_y"])
        angles["left_arm"] = Angle_converter(lengths["left_elbow_x"], lengths["left_elbow_y"],
                                             lengths["left_hand_x"], lengths["left_hand_y"])
        #print(angles)
        #This tries to tell if the technique of the excercise is good or not
        technique = pd.DataFrame(columns=['technique_shoulder', 'technique_elbow', 'technique_height'])
        technique['technique_shoulder'] = np.where(
            np.abs(angles['right_shoulder'] - angles['left_shoulder']) < tol_elbow_deg,
            'Good', 'Bad')
        technique['technique_elbow'] = np.where(np.abs(angles['right_arm'] - angles['left_arm']) < tol_elbow_deg,
                                                'Good',
                                                'Bad')
        technique['technique_height'] = np.where(
            np.abs(lengths['right_hand_y'] - lengths['left_hand_y']) < tol_hand_pos,
            'Good', 'Bad')
        technique['equal_distance_arms'] = np.where(
            np.abs(np.abs(lengths['right_shoulder_x'] - lengths['right_hand_x']) -
                   np.abs(lengths['left_shoulder_x'] - lengths['left_hand_x'])) < tol_hand_pos,
            'Good', 'Bad')
        print(technique)
######################################################################################################################
        # Draw Skeleton
        hands = [4,7]
        POSE_PAIRS.append(hands)
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA][0] is not None and points[partB][0] is not None:
                color = technique_color(technique.iloc[-1], pair) #Gets the color
                cv2.line(frame, points[partA], points[partB], color, 3, lineType=cv2.LINE_AA)
                cv2.circle(frame, points[partA], 8, (0,255,0), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0,255,0), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        del POSE_PAIRS[-1]
        # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
        # cv2.imshow('Output-Keypoints', frameCopy)
        cv2.imshow('Output-Skeleton', frame)
        vid_writer.write(frame)

    vid_writer.release()
    cv2.destroyAllWindows()
    if write_bool:
        df.to_csv('movement.csv', index=False)

    return df

if __name__ == '__main__':
    model('resources/whathth.mp4', True)
