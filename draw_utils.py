import numpy as np
from numpy import ndarray
import PIL
from PIL import ImageDraw, ImageFont
from PIL.Image import Image


#外加的
coco_colors = [(255, 0, 127), (254, 37, 103), (251, 77, 77), (248, 115, 51),
               (242, 149, 25), (235, 180, 0), (227, 205, 24), (217, 226, 50),
               (206, 242, 76), (193, 251, 102), (179, 254, 128), (165, 251, 152),
               (149, 242, 178), (132, 226, 204), (115, 205, 230), (96, 178, 255),
               (78, 149, 255), (59, 115, 255), (39, 77, 255), (18, 37, 255), (0, 0, 255)]

coco_skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                 [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                 [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                 [3, 5], [4, 6]]


# COCO 17 points
point_name = ["nose", "left_eye", "right_eye",
              "left_ear", "right_ear",
              "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow",
              "left_wrist", "right_wrist",
              "left_hip", "right_hip",
              "left_knee", "right_knee",
              "left_ankle", "right_ankle"]

point_color = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
               (240, 2, 127), (240, 2, 127),
               (255, 255, 51), (255, 255, 51),
               (254, 153, 41), (44, 127, 184),
               (217, 95, 14), (0, 0, 255),
               (255, 255, 51), (255, 255, 51), (228, 26, 28),
               (49, 163, 84), (252, 176, 243), (0, 176, 240),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142)]


def draw_keypoints(img: Image,
                   keypoints: ndarray,
                   scores: ndarray = None,
                   thresh: float = 0.2,
                   r: int = 4,
                   draw_text: bool = False,
                   font: str = 'arial.ttf',
                   font_size: int = 10):
    scale = 1/150
    thickness = min(int(img.shape[0]*scale), int(img.shape[1]*scale))
    if isinstance(img, ndarray):
        img = PIL.Image.fromarray(img)

    if scores is None:
        scores = np.ones(keypoints.shape[0])

    if draw_text:
        try:
            font = ImageFont.truetype(font, font_size)
        except IOError:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    for i, (point, score) in enumerate(zip(keypoints, scores)):
        if score > thresh and np.max(point) > 0:
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                         fill=point_color[i],
                         outline=(255, 255, 255))
            if draw_text:
                draw.text((point[0] + r, point[1] + r), text=point_name[i], font=font)

    #draw the connection
    for i, segment in enumerate(coco_skeleton):
        point1_id, point2_id = segment

        point1 = keypoints[point1_id]
        point2 = keypoints[point2_id]

        draw.line((int(point1[0]), int(point1[1])) +
                  (int(point2[0]), int(point2[1])),
                  fill=coco_colors[i], width=thickness)
    return img

