import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import HighResolutionNet
from draw_utils import draw_keypoints
import transforms

from angel_estimate import cal_angle, angle_compare



def predict_all_person():
    # TODO
    pass

def setup():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    flip_test = True
    resize_hw = (256, 192)
    standard_pose_path = "./standard/"
    weights_path = "./hrnet_w32.pth"
    keypoint_json_path = "person_keypoints.json"
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read json file
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)
    return device, flip_test, standard_pose_path, weights_path, keypoint_json_path, data_transform, person_info

def load_in_database(pic_paths,pose):
    device, flip_test, standard_pose_path, weights_path, keypoint_json_path, data_transform, person_info = setup()
    #pose = input('Which pose do you want to define?')  # 先考虑正面和侧面把   四个图片放一个文件夹  打开名为pose的文件夹？依次读入四张图片 前后左右
    pose_txt = open(standard_pose_path + pose + ".txt", mode='w+')
    # pose_txt.write(pose)
    # pose_txt.write('\n')
    #pic_path = './standard/{}/'.format(pose)
    saved = 1
    #传入两张图片
    for pic_path in pic_paths:
        img = cv2.imread(pic_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        flipped_img = cv2.flip(img, 1)                                                                                  #左右翻转，estimate时去最match的结果

        flipped_img_tensor, flipped_target = data_transform(flipped_img, {"box": [0, 0, flipped_img.shape[1] - 1, flipped_img.shape[0] - 1]})
        img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
        flipped_img_tensor = torch.unsqueeze(flipped_img_tensor, dim=0)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

        # create model
        # HRNet-W32: base_channel=32
        # HRNet-W48: base_channel=48
        model = HighResolutionNet(base_channel=32)
        weights = torch.load(weights_path, map_location=device)
        weights = weights if "model" not in weights else weights["model"]
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

        with torch.inference_mode():
            outputs = model(img_tensor.to(device))
            flipped_outputs = model(flipped_img_tensor.to(device))
            if flip_test:
                flip_tensor = transforms.flip_images(img_tensor)
                flip_outputs = torch.squeeze(
                    transforms.flip_back(model(flip_tensor.to(device)), person_info["flip_pairs"]),
                )
                # feature is not aligned, shift flipped heatmap for higher accuracy
                # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
                outputs = (outputs + flip_outputs) * 0.5

                flip_tensor1 = transforms.flip_images(flipped_img_tensor)
                flip_outputs1 = torch.squeeze(
                    transforms.flip_back(model(flip_tensor1.to(device)), person_info["flip_pairs"]),
                )
                # feature is not aligned, shift flipped heatmap for higher accuracy
                # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                flip_outputs1[..., 1:] = flip_outputs1.clone()[..., 0: -1]
                flipped_outputs = (flipped_outputs + flip_outputs1) * 0.5

            keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
            flipped_keypoints, flipped_scores = transforms.get_final_preds(flipped_outputs, [target["reverse_trans"]], True)
            keypoints = np.squeeze(keypoints)
            scores = np.squeeze(scores)
            flipped_keypoints = np.squeeze(flipped_keypoints)
            #flipped_scores = np.squeeze(flipped_scores)

            plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=3)
            angles = cal_angle(keypoints)
            flipped_angles = cal_angle(flipped_keypoints)
            for angle in angles:
                pose_txt.writelines("{}\n".format(angle))
            print('{} keypoints loaded'.format(os.path.basename(pic_path)))
            plot_img.save("./standard_samples/{}".format(os.path.basename(pic_path)))
            # plt.imshow(plot_img)
            # plt.show()
    return pose + " loaded successfully!"
def predict_new(pic_paths):
    device, flip_test, standard_pose_path, weights_path, keypoint_json_path, data_transform, person_info = setup()
    front_dict = {}
    lateral_dict = {}
    report = ""
    for file in os.listdir(standard_pose_path):  # 对路径中的每个标准动作txt进行操作
        if '.txt' not in file:
            continue
        txt = open(standard_pose_path + file, "r+")
        front = []
        lateral = []
        gesture = txt.name[11:-4]  # 正则化？去掉多余的路径字符和后缀
        for i in range(22):  # range is correspondent with number of vetcor comparisons
            angle = txt.readline()
            angle = float(angle.strip())
            if i < 11:
                front.append(angle)
            else:
                lateral.append(angle)
        front_dict[gesture] = front
        lateral_dict[gesture] = lateral
    saved = 0
    for pic_path in pic_paths:
        img = cv2.imread(pic_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        flipped_img = cv2.flip(img, 1)                                                                                  #左右翻转，estimate时去最match的结果

        flipped_img_tensor, flipped_target = data_transform(flipped_img, {"box": [0, 0, flipped_img.shape[1] - 1, flipped_img.shape[0] - 1]})
        img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
        flipped_img_tensor = torch.unsqueeze(flipped_img_tensor, dim=0)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

            # create model
            # HRNet-W32: base_channel=32
            # HRNet-W48: base_channel=48
        model = HighResolutionNet(base_channel=32)
        weights = torch.load(weights_path, map_location=device)
        weights = weights if "model" not in weights else weights["model"]
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

        with torch.inference_mode():
            outputs = model(img_tensor.to(device))
            flipped_outputs = model(flipped_img_tensor.to(device))
            if flip_test:
                flip_tensor = transforms.flip_images(img_tensor)
                flip_outputs = torch.squeeze(
                    transforms.flip_back(model(flip_tensor.to(device)), person_info["flip_pairs"]),
                )
                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
                outputs = (outputs + flip_outputs) * 0.5

                flip_tensor1 = transforms.flip_images(flipped_img_tensor)
                flip_outputs1 = torch.squeeze(
                    transforms.flip_back(model(flip_tensor1.to(device)), person_info["flip_pairs"]),
                )
                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                flip_outputs1[..., 1:] = flip_outputs1.clone()[..., 0: -1]
                flipped_outputs = (flipped_outputs + flip_outputs1) * 0.5

            keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
            flipped_keypoints, flipped_scores = transforms.get_final_preds(flipped_outputs, [target["reverse_trans"]], True)
            keypoints = np.squeeze(keypoints)
            scores = np.squeeze(scores)
            flipped_keypoints = np.squeeze(flipped_keypoints)
                #flipped_scores = np.squeeze(flipped_scores)

            plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=3)
            angles = cal_angle(keypoints)
            flipped_angles = cal_angle(flipped_keypoints)
            if is_lateral(keypoints):  # 简单判断正或侧身
                pose = angle_compare(angles, flipped_angles, lateral_dict)
                report += "lateral \n"
            else:
                pose = angle_compare(angles, flipped_angles, front_dict)
                report += "front \n"
            input = os.path.basename(pic_path)
            result = "./samples/{}".format(input)
            plot_img.save(result)  # 预测结果保存在samples文件夹中
            # print(pose)
            # print("{} 可能是 {}".format(input, min(pose, key=pose.get)))
            report += "{} 可能是\n{}".format(input, min(pose, key=pose.get))
            # plt.imshow(plot_img)
            # plt.show()
            output = min(pose, key=pose.get)
            # print(report)
            # print(result)
            return report,result,input,output

def predict_single_person(method):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    flip_test = True
    resize_hw = (256, 192)
    standard_pose_path = "./standard/"                  #关节坐标和标准图片都存在standard文件夹中 应读入到字典中
    img_path = "./testpics/"
    weights_path = "./hrnet_w32.pth"
    keypoint_json_path = "person_keypoints.json"
    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read json file
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)
        #a_list = cal_angle(keypoints)
        #method为1时，存入标准动作库    不为1时输出预测结果        考虑自动读入standard文件夹中的所有动作
    if method == '1':                                                         #先简单判断正面背面还是侧面，更改predict时四个方向的权重，自适应？根据正面（面的匹配度不是姿势）的匹配度计算出相应权重
        pose = input('Which pose do you want to define?')                   # 先考虑正面和侧面把   四个图片放一个文件夹  打开名为pose的文件夹？依次读入四张图片 前后左右
        pose_txt = open(standard_pose_path+pose+".txt", mode='w+')
        #pose_txt.write(pose)
        #pose_txt.write('\n')
        pic_path = './standard/{}/'.format(pose)
        saved = 1
    else:                                                                   #图片路径设置到待预测的路径  读入所有已定义的标准动作按朝向分别做成字典
        pic_path = img_path
        front_dict={}
        lateral_dict={}
        for file in os.listdir(standard_pose_path):                     #对路径中的每个标准动作txt进行操作
            if '.txt' not in file:
                continue
            txt = open(standard_pose_path+file, "r+")
            front = []
            lateral = []
            gesture = txt.name[11:-4]                                   #正则化？去掉多余的路径字符和后缀
            for i in range(22):                                         #range is correspondent with number of vetcor comparisons
                angle = txt.readline()
                angle = float(angle.strip())
                if i < 11:
                    front.append(angle)
                else:
                    lateral.append(angle)
            front_dict[gesture]=front
            lateral_dict[gesture]=lateral
        saved = 0


    for file in os.listdir(pic_path):
        if '.jpg' not in file:
            continue
        img = cv2.imread(pic_path + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        flipped_img = cv2.flip(img, 1)                                                                                  #左右翻转，estimate时去最match的结果

        flipped_img_tensor, flipped_target = data_transform(flipped_img, {"box": [0, 0, flipped_img.shape[1] - 1, flipped_img.shape[0] - 1]})
        img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
        flipped_img_tensor = torch.unsqueeze(flipped_img_tensor, dim=0)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

        # create model
        # HRNet-W32: base_channel=32
        # HRNet-W48: base_channel=48
        model = HighResolutionNet(base_channel=32)
        weights = torch.load(weights_path, map_location=device)
        weights = weights if "model" not in weights else weights["model"]
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

        with torch.inference_mode():
            outputs = model(img_tensor.to(device))
            flipped_outputs = model(flipped_img_tensor.to(device))
            if flip_test:
                flip_tensor = transforms.flip_images(img_tensor)
                flip_outputs = torch.squeeze(
                    transforms.flip_back(model(flip_tensor.to(device)), person_info["flip_pairs"]),
                )
                # feature is not aligned, shift flipped heatmap for higher accuracy
                # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
                outputs = (outputs + flip_outputs) * 0.5

                flip_tensor1 = transforms.flip_images(flipped_img_tensor)
                flip_outputs1 = torch.squeeze(
                    transforms.flip_back(model(flip_tensor1.to(device)), person_info["flip_pairs"]),
                )
                # feature is not aligned, shift flipped heatmap for higher accuracy
                # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
                flip_outputs1[..., 1:] = flip_outputs1.clone()[..., 0: -1]
                flipped_outputs = (flipped_outputs + flip_outputs1) * 0.5

            keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
            flipped_keypoints, flipped_scores = transforms.get_final_preds(flipped_outputs, [target["reverse_trans"]], True)
            keypoints = np.squeeze(keypoints)
            scores = np.squeeze(scores)
            flipped_keypoints = np.squeeze(flipped_keypoints)
            #flipped_scores = np.squeeze(flipped_scores)

            plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=3)
            angles = cal_angle(keypoints)
            flipped_angles = cal_angle(flipped_keypoints)
            if saved:
                for angle in angles:
                    pose_txt.writelines("{}\n".format(angle))
                print('{} keypoints loaded'.format(file))
                plot_img.save("./standard_samples/{}".format(file))
            else:
                if is_lateral(keypoints):                                           #简单判断正或侧身
                    pose = angle_compare(angles, flipped_angles, lateral_dict)
                    print("lateral")
                else:
                    pose = angle_compare(angles, flipped_angles, front_dict)
                    print("front")

                plot_img.save("./samples/{}".format(file))                          #预测结果保存在samples文件夹中
                print(pose)
                print("{} 可能是 {}".format(file, min(pose, key=pose.get)))
            plt.imshow(plot_img)
            plt.show()

            """             

                #pose = angle_compare(a_list, norm_dict)
                plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=3)
                cv2.namedWindow('image', 0)
                cv2.imshow('image', plot_img)
                cv2.imwrite("./samples/"+file, plot_img)
                #print(pose)
                #print("可能是 {}".format(min(pose, key=pose.get)))"""

def is_lateral(keypoints):
    ear_dist = np.sqrt((keypoints[3][0]-keypoints[4][0])**2+(keypoints[3][1]-keypoints[4][1])**2)
    norm_dist = (np.sqrt((keypoints[0][0]-keypoints[3][0])**2+(keypoints[0][1]-keypoints[3][1])**2)+np.sqrt((keypoints[0][0]-keypoints[4][0])**2+(keypoints[0][1]-keypoints[4][1])**2))/2
    if ear_dist < norm_dist:
        return True
    else:
        return False



if __name__ == '__main__':                                      #分set还是predict set设置标准动作 predict进行预测
    method = input('1存入标准动作库 2预测: ')
    predict_single_person(method)
