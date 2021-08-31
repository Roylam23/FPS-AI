import argparse
import threading
import cv2
import numpy as np
import mss
import torch
import mouse
import imutils
import time
import keyboard
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width
import concurrent.futures

class VideoReader():
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
    
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(grab, mon)
            future1 = executor.submit(grab, mon)
            # img = np.concatenate((future.result(), future1.result()))
            # img = cv2.add(future.result(),future1.result())
            img = future.result()
        return img

def grab(mon):
    with mss.mss() as sct:
        img = np.array(sct.grab(mon))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale,
                            fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(
        scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
    tensor_img = torch.from_numpy(padded_img).permute(
        2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()
    
    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(
        stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio,
                          fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio,
                      fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()
    x = 0
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    aim = False
    delay = 1
    for img in image_provider:
        start_time = time.time()
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(
            net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(
            all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(
                        all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[0][18])
            current_poses.append(pose)
        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        if keyboard.is_pressed('+'):
            aim = True
        if keyboard.is_pressed('-'):
            aim = False
        # if bool(current_poses):
        #     current_poses[0].draw_spec(img,aim)
        img = cv2.addWeighted(orig_img, 1.5, img, -0.5, 0)
        for pose in current_poses:
            pose.draw_spec(img,aim)
        ftime = 1.0 / (time.time() - start_time)
        fps = "FPS: "+str(round(ftime))
        cv2.putText(img, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.imshow('TEST AI1', img)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (245, 197, 66))
            # if track:
            #     cv2.putText(img, 'Target: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
            #                 cv2.FONT_HERSHEY_COMPLEX, 0.5, (234,182,118))
        # cv2.resizeWindow("TEST AI", 1200, 880);
        # img = cv2.resize(img,(int(width*2),int(height*2)),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('FPS AI', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str,default='checkpoint_iter_370000.pth',
                        required=False, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256,
                        help='network input layer height size')
    parser.add_argument('--video', type=str, default='',
                        help='path to video file or camera id')
    parser.add_argument('--cpu', action='store_true',
                        help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1,
                        help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1,
                        help='smooth pose keypoints')
    parser.add_argument('--width',type=int,default=2560)
    parser.add_argument('--height',type=int,default=1440)
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)
    print("""
        ██████╗ ██╗      █████╗ ██╗     ██╗███████╗██████╗ ███████╗██╗ 
        ██╔══██╗██║     ██╔══██╗██║    ██╔╝██╔════╝██╔══██╗██╔════╝╚██╗
        ██████╔╝██║     ███████║██║    ██║ █████╗  ██████╔╝███████╗ ██║
        ██╔══██╗██║     ██╔══██║██║    ██║ ██╔══╝  ██╔═══╝ ╚════██║ ██║
        ██║  ██║███████╗██║  ██║██║    ╚██╗██║     ██║     ███████║██╔╝
        ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝     ╚══════╝╚═╝ 
                                                                
    """)
    time.sleep(0.5)
    print('Artificial intelligence is starting...')
    frame_provider = VideoReader()
    width = int(700)
    height = int(500)
    top = int((args.height - height) / 2)
    left = int((args.width - width) / 2)
    mon = {"top": top, "left": left, "width": width, "height": height}
    run_demo(net, frame_provider, args.height_size,
             args.cpu, args.track, args.smooth)
