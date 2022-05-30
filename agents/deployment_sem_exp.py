import math
import os
import cv2
import numpy as np
import skimage.morphology
from PIL import Image
from torchvision import transforms

from ..envs.utils.fmm_planner import FMMPlanner
from ..envs.habitat.objectgoal_env import ObjectGoal_Env
from .utils.semantic_prediction import SemanticPredMaskRCNN
from ..constants import color_palette
from ..envs.utils import pose as pu
from .utils import visualization as vu
import pickle
from . semantic_gt_loader import Semantic_env
from glob import glob
import time 


import rospy
from std_msgs.msg import String



class Sem_Exp_Env_Agent():
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args):

        self.args = args
        self.thold = args.sem_pred_prob_thr
        self.previous_image = np.random.rand(480,640,3)
        #get the scene =
        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        pickle.dump(args,open('./args_example.pkl','wb'))

        self.sem_pred = SemanticPredMaskRCNN(args)

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)

        self.obs = None
        # self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.prev_state = np.array([0,0])


    def reset(self):
        args = self.args

        # self.obs_shape = obs.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None

        # return obs, info

    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # plan
        # if planner_inputs["wait"]:
        #     self.last_action = None
        #     self.info["sensor_pose"] = [0., 0., 0.]
        #     return np.zeros(self.obs.shape), 0., False, self.info

        # Reset reward if new long-term goal
        # if planner_inputs["new_goal"]:
        #     self.info["g_reward"] = 0
        # map_to_save = planner_inputs['map_pred']
        # if(self.imcounter%10 == 9):
        #     cv2.imwrite(self.imgs_dir.format(self.thold,'map',self.imcounter,'png'),map_to_save)
        action = self._plan(planner_inputs)


        # if action >= 0:
        # if action is not None:
            # act
        action = {'action': action}

        # preprocess obs
        self.last_action = action['action']


        return action['action']

        # else:
        #     self.last_action = None
        #     self.info["sensor_pose"] = [0., 0., 0.]
        #     return np.zeros(self.obs_shape), 0., False, self.info

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        goal = planner_inputs['goal']

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        # Collision check
        if self.last_action == 'MOVE_FORWARD':
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold:  # Collision
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        stg, stop = self._get_stg(map_pred, start, np.copy(goal),
                                  planning_window)

        # Deterministic Local Policy
        if stop and planner_inputs['found_goal'] == 1:
            action = 'STOP'#0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            if relative_angle > self.args.turn_angle / 2.:
                action = 'TURN_RIGHT' #3  # Right
            elif relative_angle < -self.args.turn_angle / 2.:
                action = 'TURN_LEFT' #2  # Left
            else:
                action = 'MOVE_FORWARD' #1  # Forward
            # print('chosen action = {},stg = {},current_pose = {}'.format(action,stg,start))

        return action

    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible,prev_state = self.last_loc)
        selem = skimage.morphology.disk(10)
        goal = skimage.morphology.binary_dilation(
            goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        
        stg_x, stg_y, _, stop = planner.get_short_term_goal(state,self.curr_loc)
        self.prev_state = state

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y), stop

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]

        sem_seg_pred = self._get_sem_pred(
            rgb.astype(np.uint8), use_seg=use_seg)
    
        # print('before => depth_min = {} | depth_max = {}'.format(depth.min(),depth.max())) 
        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)
        # print('before => depth_min = {} | depth_max = {}'.format(depth.max(),depth.min()))
        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1
        # print('\n\n PREPROCESSING DEPTH!!! \n\n\n\n')
        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def _get_sem_pred(self, rgb, use_seg=True):
        if use_seg:
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        # print(semantic_pred,semantic_pred.shape)
        return semantic_pred

