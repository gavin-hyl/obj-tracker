import os
from collections import namedtuple

import cv2 as cv
from openpyxl import load_workbook

all_params = {
    1: "Steel-4V-0.85m",
    2: "Steel-8V-0.85m",
    3: "Steel-12V-0.85m",
    4: "Steel-12V-0.35m",
    5: "Steel-12V-1.40m",
    6: "Ceramic-12V-0.85m",
    7: "Wood-12V-0.85m"
}

exp_set = namedtuple('exp_set', 'v1y, o1, E_ret')
averages = {
    1: exp_set(-3.8594999999999997, 1119.7975999999996, 0.7147963287663259),
    2: exp_set(-3.9825416666666666, 2406.5575833333332, 0.6423058785890958),
    3: exp_set(-4.030448275862069, 3919.5288620689653, 0.6891950995462978),
    4: exp_set(-2.6173636363636366, 4294.478318181819, 0.6586052150104104),
    5: exp_set(-5.027266666666666, 4437.422333333334, 0.7140472415342465),
    6: exp_set(-3.8116562500000004, 3978.0616250000003, 0.6310358434844984),
    7: exp_set(-4.307678571428572, 4187.489928571428, 0.2288317361800067)
}

boundary_vx_conditions = namedtuple('boundary_vx_conditions', 'angle, pos, neg')
boundary_dvx = {
    1: boundary_vx_conditions(77.04723548662449, 0.6089989295175774, -0.7626596194209995),
    2: boundary_vx_conditions(77.04723548662449, 0.5433764082151202, -0.8736222815791929),
    3: boundary_vx_conditions(77.04723548662449, 0.449744366538, -0.9876567739387162),
    4: boundary_vx_conditions(77.04723548662449, 0.17521177913028538, -0.7647266266198801),
    5: boundary_vx_conditions(77.04723548662449, 0.5913647339675986, -1.2003377199225083),
    6: boundary_vx_conditions(71.89225525772078, 0.6605124709941809, -1.194956869415346),
    7: boundary_vx_conditions(61.05740620691493, 1.0280192670182613, -1.5098731027665429),
}

COFs = {
    1: 0.231,
    2: 0.231,
    3: 0.231,
    4: 0.231,
    5: 0.231,
    6: 0.327,
    7: 0.553,
}


def rename(folder):
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"{count + 1}.mp4"
        src = f"{folder}/{filename}"
        dst = f"{folder}/{dst}"
        os.rename(src, dst)


def snapshots(vid_path, check, flip, reb=0):
    vid = cv.VideoCapture(vid_path)
    raw_frames = []
    success, img = vid.read()
    while success:
        raw_frames.append(img)
        success, img = vid.read()

    crop_top = 0
    crop_bottom = 0
    crop_left = 0
    crop_right = 0

    img = raw_frames[0]
    scale_percent = 40
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    for i, _ in enumerate(raw_frames):
        resized = cv.resize(_, dim)
        if flip:
            resized = cv.flip(resized, -1)
        cropped = resized[crop_top:dim[1] - crop_bottom, crop_left:dim[0] - crop_right]
        if check:
            cv.putText(cropped, f"frame: {i}", org=(10, 100), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,
                       color=(255, 255, 255), thickness=1, lineType=2)
            cv.imshow('vid', cropped)
            cv.waitKey(0)
        else:
            if abs(reb-i) % 30 == 0:
                cv.putText(cropped, f"frame: {i-(reb%30)}", org=(10, 100), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=1, color=(255, 255, 255), thickness=1, lineType=2)
                cv.imshow('vid', cropped)
                cv.waitKey(1)
                cv.imwrite(f"Videos/Trimmed/shot {i-(reb%30)}.jpg", cropped)


if __name__ == "__main__":
    print("hi")
