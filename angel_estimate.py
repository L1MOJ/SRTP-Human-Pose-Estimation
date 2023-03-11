import numpy as np

def cal_angle(keypoints):
    main_key = keypoints[5:]
    all_angle = [[[0, 2], [2, 4]], [[1, 3], [3, 5]], [[0, 1], [1, 3]], [[1, 0], [0, 2]],
                 [[7, 6], [6, 8]], [[6, 7], [7, 9]], [[7, 9], [9, 11]], [[6, 8], [8, 10]],
                 [[0, 6], [0, 2]], [[1, 7], [1, 3]], [[0, 2], [1, 3]]]
    angle_list = []
    for angle in all_angle:

        v1 = [main_key[angle[0][1]][0] - main_key[angle[0][0]][0],               #v1 x
              main_key[angle[0][1]][1] - main_key[angle[0][0]][1]]               #v1 y
        v2 = [main_key[angle[1][1]][0] - main_key[angle[1][0]][0],                #v2 x
              main_key[angle[1][1]][1] - main_key[angle[1][0]][1]]                #v2 y
        angle = dot_product_angle(v1, v2)
        angle_list.append(angle)
    return angle_list

def angle_compare(a_angle, flipped_a_angle, norm_angle):                                #将待测图像反转，去最接近的结果
    difference, flipped_difference = 0, 0
    pre_softmax = []
    name = []
    for key in norm_angle:
        for i in range(0, len(a_angle)):
            difference += (a_angle[i] - norm_angle[key][i])**2
            flipped_difference += (flipped_a_angle[i] - norm_angle[key][i])**2
        difference = np.sqrt(min(difference, flipped_difference))
        name.append(key)
        pre_softmax.append(difference)
    #pre_softmax -= np.max(pre_softmax)
    differ = np.exp(pre_softmax) / np.sum(np.exp(pre_softmax))
    dif_group = dict(zip(name, differ))
    return dif_group



def dot_product_angle(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos)
        return angle
    return 0
