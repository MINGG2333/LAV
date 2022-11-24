import os
from collections import deque

import numpy as np
import math


DEBUG = int(os.environ.get('HAS_DISPLAY', 0))


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    
    EARTH_RADIUS = 6371e3 # 6371km
    
    def __init__(self, global_plan, curr_threshold=20, next_threshold=75, debug_size=720):
        self.route = deque()
        self.route_cmd = []
        self.curr_threshold = curr_threshold
        self.next_threshold = next_threshold

        # Convert lat,lon to x,y
        cos_0 = 0.
        for gnss, _ in global_plan:
            cos_0 += gnss['lat'] * (math.pi / 180)
        cos_0 = cos_0 / (len(global_plan))
        self.cos_0 = cos_0

        # like def set_route() as this implement rm self.route.popleft()
        for node in global_plan:
            gnss, cmd = node

            x, y = self.latlon_to_xy(gnss['lat'], gnss['lon'])
            self.route.append((x, y))
            self.route_cmd.append(cmd)

        self.debug = Plotter(debug_size)
        self.centre = None
        self.store_wps = []
        self.store_wps_real = []

        self.current_idx = 0
        self.checkpoint = self.route[0]

        self.centre = np.array(self.route[int(len(self.route)//2)])

    def run_step(self, gnss):

        x, y = self.latlon_to_xy(gnss[0], gnss[1]) # jxy: like [:2] and _get_position
        gps = np.array([x,y])

        self.debug.clear()

        self.cur_veh_gps = gps
        if self.centre is None:
            self.centre = gps
        
        wx, wy = np.array(self.checkpoint)
        curr_distance = np.linalg.norm([wx-x, wy-y])

        for i, (wx, wy) in enumerate(self.route):
            
            distance = np.linalg.norm([wx-x, wy-y])
            
            if distance < self.next_threshold and i - self.current_idx==1 and curr_distance < self.curr_threshold:
                self.checkpoint = [wx, wy]
                self.current_idx += 1
                break

        for i, route_pt in enumerate(self.route):
            if i == self.current_idx - 1:
                self.debug.dot(self.centre, np.array(route_pt), (0, 255, 0))
            elif i < self.current_idx:
                continue
            elif i == self.current_idx:
                self.debug.dot(self.centre, np.array(route_pt), (255, 0, 0))
            else:
                r = 255 * int(True)
                g = 255 * int(self.route_cmd[i].value == 4)
                b = 255
                self.debug.dot(self.centre, np.array(route_pt), (r, g, b))

        self.store_wps_real.append(gps)
        for pos in self.store_wps_real:
            self.debug.dot(self.centre, pos, (0, 0, 255))

        return np.array(self.checkpoint) - [x,y], self.route_cmd[self.current_idx], gps


    def latlon_to_xy(self, lat, lon):

        x = self.EARTH_RADIUS * lat * (math.pi / 180)
        y = self.EARTH_RADIUS * lon * (math.pi / 180) * math.cos(self.cos_0)

        return x, y

    def run_step2(self, _poses, is_gps=True, color=(255, 255, 0), store=True):
        if is_gps:
            poses = _poses.copy()
        else:
            poses = _poses + self.cur_veh_gps if len(_poses) else []

        for pos in self.store_wps:
            self.debug.dot(self.centre, pos, color)

        if store:
            self.store_wps.extend(list(poses))

        for pos in poses:
            self.debug.dot(self.centre, pos, color)

    def show_route(self):
        self.debug.show()
