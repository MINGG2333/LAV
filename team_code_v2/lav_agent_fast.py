import os
import math
import yaml
import json
import time
from PIL import Image
import numpy as np
import wandb
import cv2
import torch
import carla
import random
import matplotlib.cm

from collections import deque
from torch.nn import functional as F

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from team_code_v2_models.lidar import LiDARModel
from team_code_v2_models.uniplanner import UniPlanner
from team_code_v2_models.bev_planner import BEVPlanner
from team_code_v2_models.rgb import RGBSegmentationModel, RGBBrakePredictionModel
from pid import PIDController
from ekf import EKF
from point_painting import CoordConverter, point_painting
from planner import RoutePlanner
from waypointer import Waypointer
from model_inference import InferModel

# jxy: addition; (add display.py and fix RoutePlanner.py)
from team_code.display import HAS_DISPLAY, Saver, debug_display
# addition from team_code/map_agent.py
from carla_project.src.common import CONVERTER, COLOR
from carla_project.src.carla_env import draw_traffic_lights, get_nearby_lights


def get_entry_point():
    return 'LAVAgent'

NUM_REPEAT = 4
GAP = NUM_REPEAT + 1
FPS = 20.
PIXELS_PER_METER = 4

CAMERA_YAWS = [-60,0,60]

class LAVAgent(AutonomousAgent):
    def sensors(self):
        sensors = [
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0., 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.other.imu',  'x': 0., 'y': 0., 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,'sensor_tick': 0.05, 'id': 'IMU'},
            # jxy: addition from team_code/map_agent.py
            {
                'type': 'sensor.camera.semantic_segmentation',
                'x': 0.0, 'y': 0.0, 'z': 100.0,
                'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                'width': 512, 'height': 512, 'fov': 5 * 10.0,
                'id': 'map'
                },
        ]

        # Add LiDAR
        sensors.append({
            'type': 'sensor.lidar.ray_cast', 'x': 0.0, 'y': 0.0, 'z': self.camera_z, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 
            'id': 'LIDAR'
        })

        # Add cameras
        for i, yaw in enumerate(CAMERA_YAWS):
            sensors.append({'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0.0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 256, 'height': 288, 'fov': 64, 'id': f'RGB_{i}'})

        sensors.append({'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0.0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 480, 'height': 288, 'fov': 40, 'id': 'TEL_RGB'})

        return sensors

    def setup(self, path_to_conf_file):

        self.track = Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        return AgentSaver

        # jxy: add return AgentSaver and init_ads (setup keep 5 lines); rm save_path;
    def init_ads(self, path_to_conf_file):

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        self.device = torch.device('cuda')

        self.waypointer = None
        self.planner    = None

        wandb.init(project='lav_eval')

        # Setup models
        self.lidar_model = LiDARModel(
            num_input=len(self.seg_channels)+10+self.num_frame_stack if self.point_painting else 10,
            backbone=self.backbone,
            num_features=self.num_features,
            min_x=self.min_x, max_x=self.max_x,
            min_y=self.min_y, max_y=self.max_y,
            pixels_per_meter=self.pixels_per_meter,
        ).to(self.device)

        bev_planner = BEVPlanner(
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size,
            feature_x_jitter=self.feature_x_jitter,
            feature_angle_jitter=self.feature_angle_jitter,
            x_offset=0, y_offset=1+self.min_x/((self.max_x-self.min_x)/2),
            num_cmds=self.num_cmds,
            num_plan=self.num_plan,
            num_plan_iter=self.num_plan_iter,
            num_frame_stack=self.num_frame_stack,
        ).to(self.device)

        self.uniplanner = UniPlanner(
            bev_planner,
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size,
            feature_x_jitter=self.feature_x_jitter,
            feature_angle_jitter=self.feature_angle_jitter,
            x_offset=0, y_offset=1+self.min_x/((self.max_x-self.min_x)/2),
            num_cmds=self.num_cmds,
            num_plan=self.num_plan,
            num_input_feature=self.num_features[-1]*6,
            num_plan_iter=self.num_plan_iter,
        ).to(self.device)

        self.bra_model = torch.jit.load(self.bra_model_trace_dir)
        self.seg_model = torch.jit.load(self.seg_model_trace_dir)

        # Load the models
        self.lidar_model.load_state_dict(torch.load(self.lidar_model_dir))
        self.uniplanner.load_state_dict(torch.load(self.uniplanner_dir))

        self.lidar_model.eval()
        self.uniplanner.eval()
        self.bra_model.eval()
        self.seg_model.eval()

        self.infer_model = InferModel(self.lidar_model, self.uniplanner, self.camera_x, self.camera_z).to(self.device)

        # Coordinate converters for point-painting
        self.coord_converters = [CoordConverter(
            cam_yaw, lidar_xyz=[0,0,self.camera_z], cam_xyz=[self.camera_x,0,self.camera_z],
            rgb_h=288, rgb_w=256, fov=64
        ) for cam_yaw in CAMERA_YAWS]

        # Setup tracker TODO: update 1 to actual cos0
        self.ekf = EKF(1, 1.477531, 1.393600)
        self.ekf_initialized = False

        # FIFO
        self.lidars = deque([])
        self.locs = deque([])
        self.oris = deque([])

        # Book-keeping
        self.vizs = []
        self.num_frames = 0

        self.prev_lidar = None
        self.num_frame_keep = (self.num_frame_stack + 1) * GAP

        self.turn_controller = PIDController(K_P=self.turn_KP, K_I=self.turn_KI, K_D=self.turn_KD, n=self.turn_n)
        self.speed_controller = PIDController(K_P=self.speed_KP, K_I=self.speed_KI, K_D=self.speed_KD, n=self.speed_n)

        self.lane_change_counter = 0
        self.stop_counter = 0
        self.force_move = 0
        self.lane_changed = None

    def flush_data(self):

        if self.log_wandb:
            wandb.log({
                'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
            })

        self.vizs.clear()

    def destroy(self):

        if len(self.vizs) == 0:
            return

        self.flush_data()

        self.waypointer = None
        self.planner    = None
        self.prev_lidar = None
        self.coord_converters = None
        self.turn_controller = None
        self.speed_controller = None

        self.num_frames = 0

        self.lane_change_counter = 0
        self.stop_counter = 0
        self.force_move = 0
        self.lane_changed = None

        self.ekf = None
        self.lidars.clear()
        self.locs.clear()
        self.oris.clear()

        self.ekf_initialized = False

        torch.cuda.empty_cache()

        super().destroy()

    def _init(self):

        del self.lidar_model
        del self.uniplanner
        del self.bra_model
        del self.seg_model

        self.initialized = True

        super()._init() # jxy add

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        self.num_frames += 1

        _, lidar = input_data.get('LIDAR')
        _, gps   = input_data.get('GPS')
        _, imu   = input_data.get('IMU')
        _, ego   = input_data.get('EGO')
        spd      = ego.get('speed')

        compass = imu[-1]
        
        # Let's hope this only happens when compass == 0 or 2pi
        # https://discord.com/channels/444206285647380482/551506571608326156/769089544103919626
        if np.isnan(compass):
            compass = 0.

        if not self.ekf_initialized:
            self.ekf.init(*gps[:2], compass-math.pi/2)
            self.ekf_initialized = True

        loc, ori = self.ekf.x[:2], self.ekf.x[2]

        if spd < 0.1:
            self.stop_counter += 1
        else:
            self.stop_counter = 0

        lidar = torch.tensor(lidar, dtype=torch.float, device=self.device)

        if self.num_frames <= 1:
            self.prev_lidar = lidar
            return carla.VehicleControl()


        if self.prev_lidar is not None:
            cur_lidar = torch.cat([lidar, self.prev_lidar])
        else:
            cur_lidar = lidar

        self.prev_lidar = lidar

        cur_lidar = self.preprocess(cur_lidar)

        # Paint the lidars
        rgbs = []

        for i in range(len(CAMERA_YAWS)):
            _, rgb = input_data.get(f'RGB_{i}')
            rgbs.append(rgb[...,:3][...,::-1])

        rgb = np.concatenate(rgbs, axis=1)
        all_rgb = np.stack(rgbs, axis=0)

        _, tel_rgb = input_data.get('TEL_RGB')
        tel_rgb = tel_rgb[...,:3][...,::-1].copy()
        tel_rgb = tel_rgb[:-self.crop_tel_bottom]

        all_rgbs = torch.tensor(all_rgb).permute(0,3,1,2).float().to(self.device)
        pred_sem = torch.softmax(self.seg_model(all_rgbs), dim=1)

        fused_lidar = self.infer_model.forward_paint(cur_lidar, pred_sem)

        # EKF updates and bookeepings
        self.lidars.append(fused_lidar)
        self.locs.append(loc)
        self.oris.append(ori)
        if len(self.lidars) > self.num_frame_keep:
            self.lidars.popleft()
            self.locs.popleft()
            self.oris.popleft()

        lidar_points = self.get_stacked_lidar()

        # High-level commands
        if self.waypointer is None:

            self.waypointer = Waypointer(
                self._global_plan, gps, pop_lane_change=True
            )

            self.planner = RoutePlanner(self._global_plan)
        
        _, _, cmd = self.waypointer.tick(gps)
        (wx, wy), next_cmd, pos = self.planner.run_step(gps)

        cmd_value = cmd.value - 1
        cmd_value = 3 if cmd_value < 0 else cmd_value
        
        if cmd_value in [4,5]:
            if self.lane_changed is not None and cmd_value != self.lane_changed:
                self.lane_change_counter = 0

            self.lane_change_counter += 1
            self.lane_changed = cmd_value if self.lane_change_counter > {4:300,5:300}.get(cmd_value) else None
        else:
            self.lane_change_counter = 0
            self.lane_changed = None

        if cmd_value == self.lane_changed:
            cmd_value = 3

        # Transform to ego-coordinates
        (wx, wy), R = _rotate(wx, wy, -imu[-1]+np.pi/2)

        # Predict brakes from images
        rgbs = torch.tensor(rgb[None]).permute(0,3,1,2).float().to(self.device)
        tel_rgbs= torch.tensor(tel_rgb[None]).permute(0,3,1,2).float().to(self.device)

        nxps       = torch.tensor([-wx,-wy]).float().to(self.device)

        # Motion forecast & planning
        ego_embd, ego_plan_locs, ego_cast_locs, other_cast_locs, other_cast_cmds, pred_bev, det = self.infer_model(lidar_points, nxps, cmd_value)
        ego_plan_locs = to_numpy(ego_plan_locs)
        ego_cast_locs = to_numpy(ego_cast_locs)
        other_cast_locs = to_numpy(other_cast_locs)
        other_cast_cmds = to_numpy(other_cast_cmds)

        pred_bra = self.bra_model(rgbs, tel_rgbs)

        if cmd_value in [4,5]:
            ego_plan_locs = ego_cast_locs

        if not np.isnan(ego_plan_locs).any():
            steer, throt, brake = self.pid_control(ego_plan_locs, spd, cmd_value)
        else:
            steer, throt, brake = 0, 0, 0

        if not np.isnan(ego_plan_locs).any():
            steer, throt, brake = self.pid_control(ego_plan_locs, spd, cmd_value)
        else:
            steer, throt, brake = 0, 0, 0

        self.ekf.step(spd, steer, *gps[:2], compass-math.pi/2)

        if float(pred_bra) > 0.1:
            throt, brake = 0, 1
        elif self.plan_collide(ego_plan_locs, other_cast_locs, other_cast_cmds):
            throt, brake = 0, 1
        if spd * 3.6 > self.max_speed:
            throt = 0

        if self.stop_counter >= 600: # Creep forward
            self.force_move = 20

        if self.force_move > 0:
            throt, brake = max(0.4, throt), 0
            self.force_move -= 1

        viz = self.visualize(rgb, tel_rgb, lidar_points, float(pred_bra), to_numpy(torch.sigmoid(pred_bev[0])), ego_plan_locs, other_cast_locs, other_cast_cmds, det, [-wx, -wy], cmd_value, spd, steer, throt, brake)
        self.vizs.append(viz)

        if len(self.vizs) >= 12000:
            self.flush_data()

        control = carla.VehicleControl(steer=steer, throttle=throt, brake=brake)

        # jxy addition:
        self.step = self.num_frames
        tick_data = {
                'rgb': rgb,
                'lidar': lidar,
                'gps': pos,
                'speed': spd,
                'compass': compass,
                }
        tick_data['far_command'] = next_cmd
        tick_data['R_pos_from_head'] = R
        tick_data['offset_pos'] = np.array([pos[0], pos[1]])
        # from team_code/map_agent.py:
        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'))
        topdown = input_data['map'][1][:, :, 2]
        topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights)
        tick_data['topdown'] = COLOR[CONVERTER[topdown]]

        if HAS_DISPLAY: # jxy: change
            debug_display(tick_data, control.steer, control.throttle, control.brake, self.step)

        self.record_step(tick_data, control, ego_plan_locs) # jxy: add
        return control

    # jxy: add record_step
    def record_step(self, tick_data, control, pred_waypoint=[]):
        # draw pred_waypoint
        if len(pred_waypoint):
            # pred_waypoint[:,1] *= -1
            pred_waypoint = tick_data['R_pos_from_head'].dot(pred_waypoint.T).T
        self.planner.run_step2(pred_waypoint, is_gps=False, store=False) # metadata['wp_1'] relative to ego head (as y)
        # addition: from leaderboard/team_code/auto_pilot.py
        speed = tick_data['speed']
        self._recorder_tick(control) # trjs
        ego_bbox = self.gather_info() # metrics
        self.planner.run_step2(ego_bbox + tick_data['offset_pos'], is_gps=True, store=False)
        self.planner.show_route()
        if self.save_path is not None and self.step % self.record_every_n_step == 0:
            self.save(control.steer, control.throttle, control.brake, tick_data)


    def get_stacked_lidar(self):

        loc0, ori0 = self.locs[-1], self.oris[-1]

        rel_lidars = []
        for i, t in enumerate(range(len(self.lidars)-1, -1, -GAP)):
            loc, ori = self.locs[t], self.oris[t]
            lidar = self.lidars[t]

            lidar_xyz = lidar[:,:3]
            lidar_f = lidar[:,3:]

            lidar_xyz = move_lidar_points(lidar_xyz, loc - loc0, ori0, ori)
            lidar_t = torch.zeros((len(lidar_xyz), self.num_frame_stack+1), dtype=lidar_xyz.dtype, device=self.device)
            lidar_t[:,i] = 1      # Be extra careful on this.

            rel_lidar = torch.cat([lidar_xyz, lidar_f, lidar_t], dim=-1)

            rel_lidars.append(rel_lidar)

        return torch.cat(rel_lidars)

    def plan_collide(self, ego_plan_locs, other_cast_locs, other_cast_cmds, dist_threshold_static=1.0, dist_threshold_moving=2.5):
        # TODO: Do a proper occupancy map?
        for other_trajs, other_cmds in zip(other_cast_locs, other_cast_cmds):
            init_x, init_y = other_trajs[0,0]
            if init_y > 0.5*self.pixels_per_meter:
                continue
            for other_traj, other_cmd in zip(other_trajs, other_cmds):
                if other_cmd < self.cmd_thresh:
                    continue

                spd = np.linalg.norm(other_traj[1:]-other_traj[:-1], axis=-1).mean()
                dist_threshold = dist_threshold_static if spd < self.brake_speed else dist_threshold_moving
                dist = np.linalg.norm(other_traj-ego_plan_locs, axis=-1).min() # TODO: outer norm?
                if dist < dist_threshold:
                    return True

        return False


    def pid_control(self, waypoints, speed, cmd):

        waypoints = np.copy(waypoints) * self.pixels_per_meter
        waypoints[:,1] *= -1

        # desired_speed = np.linalg.norm(waypoints[1:]-waypoints[:-1], axis=1).mean()
        # desired_speed = np.mean((waypoints[1:]-waypoints[:-1])@[0,1])
        desired_speed = np.linalg.norm(waypoints[1:]-waypoints[:-1], axis=1).mean()

        aim = waypoints[self.aim_point[cmd]]
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        # Below: experimental
        # steer = steer if desired_speed > self.brake_speed * self.pixels_per_meter * 2 else 0.

        brake = desired_speed < self.brake_speed * self.pixels_per_meter
        delta = np.clip(desired_speed * self.speed_ratio[cmd] - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        return float(steer), float(throttle), float(brake)


    def det_inference(self, heatmaps, sizemaps, orimaps, **kwargs):

        dets = []
        for i, c in enumerate(heatmaps):
            det = []
            for s, x, y in extract_peak(c, **kwargs):
                w, h = float(sizemaps[0,y,x]),float(sizemaps[1,y,x])
                cos, sin = float(orimaps[0,y,x]), float(orimaps[1,y,x])
                
                if i==1 and w < 0.1*self.pixels_per_meter or h < 0.2*self.pixels_per_meter:
                    continue
                
                # TODO: remove hardcode
                if np.linalg.norm([x-160,y-280]) <= 2:
                    continue

                det.append((x,y,w,h,cos,sin))
            dets.append(det)
        
        return dets

    def preprocess(self, lidar_xyzr, lidar_painted=None):

        idx = (lidar_xyzr[:,0] > -2.4)&(lidar_xyzr[:,0] < 0)&(lidar_xyzr[:,1]>-0.8)&(lidar_xyzr[:,1]<0.8)&(lidar_xyzr[:,2]>-1.5)&(lidar_xyzr[:,2]<-1)

        if lidar_painted is None:
            return lidar_xyzr[~idx]
        else:
            return lidar_xyzr[~idx], lidar_painted[~idx]

    def visualize(self, rgb, tel_rgb, lidar, pred_bra, pred_bev, pred_loc, cast_locs, cast_cmds, det, tgt, cmd, spd, steer, throt, brake, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)):
        lidar_viz = lidar_to_bev(
            to_numpy(lidar),
            min_x=self.min_x, max_x=self.max_x,
            min_y=self.min_y, max_y=self.max_y,
            pixels_per_meter=self.pixels_per_meter,
        ).astype(np.uint8)

        lidar_viz = cv2.cvtColor(lidar_viz, cv2.COLOR_GRAY2RGB)

        h, w, _ = lidar_viz.shape
        ego = [160,280] # TODO: compute this analytically

        for loc in pred_loc:
            cv2.circle(lidar_viz, tuple((ego+loc*self.pixels_per_meter).astype(int)), 1, (255,0,0), -1)

        cmap = matplotlib.cm.get_cmap('jet')
        for trajs, cmds in zip(cast_locs, cast_cmds):
            for traj, score in zip(trajs, cmds):
                if score < self.cmd_thresh:
                    continue
                (r,g,b,_) = cmap(score)

                for loc in traj:
                    cv2.circle(lidar_viz, tuple((ego+loc*self.pixels_per_meter).astype(int)), 1, (int(r*255),int(g*255),int(b*255)), -1)

        for x, y, ww, hh, cos, sin in det[1]:

            p1 = tuple(([x,y] + [-ww,-hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
            p2 = tuple(([x,y] + [-ww, hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
            p3 = tuple(([x,y] + [ ww, hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))
            p4 = tuple(([x,y] + [ ww,-hh]@np.array([[-sin,cos],[-cos,-sin]])).astype(int))

            cv2.drawContours(lidar_viz, np.array([[p1,p2,p3,p4]]), 0, (255,0,0), 2)

        cv2.circle(lidar_viz, tuple(np.clip(ego+np.array(tgt)*self.pixels_per_meter, 0, 255).astype(int)), 2, (0,255,0), -1)

        canvas = np.concatenate([
            cv2.resize(rgb, dsize=(int(rgb.shape[1]/rgb.shape[0]*h), h)),
            cv2.resize(tel_rgb, dsize=(int(tel_rgb.shape[1]/tel_rgb.shape[0]*h), h)),
            lidar_viz,
            cv2.cvtColor((255*pred_bev.mean(axis=0)).astype(np.uint8), cv2.COLOR_GRAY2RGB),
        ], axis=1)

        canvas = cv2.resize(canvas, dsize=(int(canvas.shape[1]/2),int(canvas.shape[0]/2)))
        
        cv2.putText(canvas, f'speed: {spd:.3f}m/s', (4, 10), *text_args)
        cv2.putText(canvas, 'cmd: {}'.format({0:'left',1:'right',2:'straight',3:'follow',4:'change left',5:'change right'}.get(cmd)), (4, 30), *text_args)
        cv2.putText(
            canvas, 
            f'steer: {steer:.3f} throttle: {throt:.3f} brake: {brake:.3f}',
            (4,20), *text_args
        )
        cv2.putText(
            canvas, 
            f'predicted brake: {pred_bra:.3f}',
            (4,40), *text_args
        )

        return canvas

def _rotate(x, y, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    return R @ [x, y], R

def to_numpy(x):
    return x.detach().cpu().numpy()

def extract_peak(heatmap, max_pool_ks=7, min_score=0.1, max_det=15, break_tie=False):
    
    # Credit: Prof. Philipp Krähenbühl in CS394D

    if break_tie:
        heatmap = heatmap + 1e-7*torch.randn(*heatmap.size(), device=heatmap.device)

    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks//2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)

    return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
            for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]

def move_lidar_points(lidar, dloc, ori0, ori1):


    dloc = dloc @ [
        [ np.cos(ori0), -np.sin(ori0)],
        [ np.sin(ori0), np.cos(ori0)]
    ]

    ori = ori1 - ori0
    lidar = lidar @ torch.tensor([
        [np.cos(ori), np.sin(ori),0],
        [-np.sin(ori), np.cos(ori),0],
        [0,0,1],
    ], dtype=torch.float, device=lidar.device)

    lidar[:,0] += dloc[0]
    lidar[:,1] += dloc[1]
    
    return lidar

def lidar_to_bev(lidar, min_x=-10,max_x=70,min_y=-40,max_y=40, pixels_per_meter=4, hist_max_per_pixel=10):
    xbins = np.linspace(
        min_x, max_x+1,
        (max_x - min_x) * pixels_per_meter + 1,
    )
    ybins = np.linspace(
        min_y, max_y+1,
        (max_y - min_y) * pixels_per_meter + 1,
    )

    hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel

    overhead_splat = hist / hist_max_per_pixel * 255.
    return overhead_splat[::-1,:]


# jxy: mv save in AgentSaver
class AgentSaver(Saver):
    def __init__(self, path_to_conf_file, dict_, list_):
        self.config_path = path_to_conf_file

        # jxy: according to sensor
        self.rgb_list = ['rgb', 'topdown', ] # 'bev', 
        self.add_img = [] # 'flow', 'out', 
        self.lidar_list = [] # 'lidar_0', 'lidar_1',
        self.dir_names = self.rgb_list + self.add_img + self.lidar_list + ['pid_metadata']

        super().__init__(dict_, list_)

    def run(self): # jxy: according to init_ads

        super().run()

    def _save(self, tick_data):    
        # addition
        # save_action_based_measurements = tick_data['save_action_based_measurements']
        self.save_path = tick_data['save_path']
        if not (self.save_path / 'ADS_log.csv' ).exists():
            # addition: generate dir for every total_i
            self.save_path.mkdir(parents=True, exist_ok=True)
            for dir_name in self.dir_names:
                (self.save_path / dir_name).mkdir(parents=True, exist_ok=False)

            # according to self.save data_row_list
            title_row = ','.join(
                ['frame_id', 'far_command', 'speed', 'steering', 'throttle', 'brake',] + \
                self.dir_names
            )
            with (self.save_path / 'ADS_log.csv' ).open("a") as f_out:
                f_out.write(title_row+'\n')

        self.step = tick_data['frame']
        self.save(tick_data['steer'],tick_data['throttle'],tick_data['brake'], tick_data)

    # addition: modified from leaderboard/team_code/auto_pilot.py
    def save(self, steer, throttle, brake, tick_data):
        # frame = self.step // 10
        frame = self.step

        # 'gps' 'thetas'
        pos = tick_data['gps']
        speed = tick_data['speed']
        far_command = tick_data['far_command']
        data_row_list = [frame, far_command.name, speed, steer, throttle, brake,]

        if frame > 1: # jxy: according to run_step
            # images
            for rgb_name in self.rgb_list + self.add_img:
                path_ = self.save_path / rgb_name / ('%04d.png' % frame)
                Image.fromarray(tick_data[rgb_name]).save(path_)
                data_row_list.append(str(path_))
            # lidar
            for i, rgb_name in enumerate(self.lidar_list):
                path_ = self.save_path / rgb_name / ('%04d.png' % frame)
                Image.fromarray(matplotlib.cm.gist_earth(tick_data['lidar_processed'][0][0, i], bytes=True)).save(path_)
                data_row_list.append(str(path_))

            # pid_metadata
            pid_metadata = tick_data['pid_metadata']
            path_ = self.save_path / 'pid_metadata' / ('%04d.json' % frame)
            outfile = open(path_, 'w')
            json.dump(pid_metadata, outfile, indent=4)
            outfile.close()
            data_row_list.append(str(path_))

        # collection
        data_row = ','.join([str(i) for i in data_row_list])
        with (self.save_path / 'ADS_log.csv' ).open("a") as f_out:
            f_out.write(data_row+'\n')

