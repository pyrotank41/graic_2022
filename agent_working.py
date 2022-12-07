
import numpy as np
import argparse
import time

import carla
from enum import Enum
from queue import PriorityQueue
import numpy as np
from math import sqrt

import matplotlib.pyplot as plt
import math
import threading

flag = 0



def prune_path(path):
        def point(p):
            return np.array([p[0], p[1], 1.]).reshape(1, -1)

        def collinearity_check(p1, p2, p3, epsilon=0.1):
            m = np.concatenate((p1, p2, p3), 0)
            det = np.linalg.det(m)
            return abs(det) < epsilon
        pruned_path = []
        # TODO: prune the path!
        p1 = path[0]
        p2 = path[1]
        pruned_path.append(p1)
        for i in range(2,len(path)):
            p3 = path[i]
            if collinearity_check(point(p1),point(p2),point(p3)):
                p2 = p3
            else:
                pruned_path.append(p2)
                p1 = p2
                p2 = p3
        pruned_path.append(p3)

        return np.array(pruned_path)

plot_shown = False
def plot(path, grid, start, goal, obstacle_occupancy, tvec, scale):
    
    global plot_shown
    plt.clf()
    path = np.array(path)/scale + tvec
    grid_x_y = np.array(np.where(grid == 1)).astype(np.int16)
    grid_x_y = grid_x_y/scale + tvec.reshape(2,1).astype(np.int16)
    start = np.array(start)/scale + tvec
    goal = np.array(goal)/scale + tvec
    
    # if obstacle_occupancy is not None:
    #     obstacle_occupancy_x_y = obstacle_occupancy/scale + tvec
    #     for i in obstacle_occupancy_x_y:
    #         plt.plot(i[0], i[1], 'o', color='red')

    plt.scatter(grid_x_y[0], grid_x_y[1], s=5, color='k', alpha=.5)
    plt.plot(start[0], start[1], 'x')
    plt.plot(goal[0], goal[1], 'xr')
    pp = np.array(path)
    plt.plot(pp[:, 0], pp[:, 1], 'g', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.pause(0.001)
    

def scaled_points_on_grid(points, scale): 
    return np.floor(points * scale).astype(int)

def get_transform_min(left_lane_location, right_lane_location, current_position, waypoint, obstacle_centers, obstacle_radius):
    # min and max values for each parameter
    if obstacle_centers is not None: 
        obstacle_centers = obstacle_centers - obstacle_radius
        min_x = min(left_lane_location[:, 0].min(), right_lane_location[:, 0].min(), current_position[0], waypoint[0], obstacle_centers[:, 0].min())
        min_y = min(left_lane_location[:, 1].min(), right_lane_location[:, 1].min(), current_position[1], waypoint[1], obstacle_centers[:, 1].min())
    else:
        min_x = min(left_lane_location[:, 0].min(), right_lane_location[:, 0].min(), current_position[0], waypoint[0])
        min_y = min(left_lane_location[:, 1].min(), right_lane_location[:, 1].min(), current_position[1], waypoint[1])
    return (min_x, min_y)

def get_transform_max(left_lane_location, right_lane_location, current_position, waypoint, obstacle_centers, obstacle_radius):
    # min and max values for each parameter
    if obstacle_centers is not None:
        obstacle_centers = obstacle_centers + obstacle_radius
        max_x = max(left_lane_location[:, 0].max(), right_lane_location[:, 0].max(), current_position[0], waypoint[0], obstacle_centers[:, 0].max())
        max_y = max(left_lane_location[:, 1].max(), right_lane_location[:, 1].max(), current_position[1], waypoint[1], obstacle_centers[:, 1].max())
    else:
        max_x = max(left_lane_location[:, 0].max(), right_lane_location[:, 0].max(), current_position[0], waypoint[0])
        max_y = max(left_lane_location[:, 1].max(), right_lane_location[:, 1].max(), current_position[1], waypoint[1])
    return (max_x, max_y)

def create_occupancy_grid_via_corners(grid, corners):
    # x = np.arange(corners[0][0]-padding, corners[2][0]+padding+1)
    # y = np.arange(corners[0][1]-padding, corners[2][1]+padding+1)
    # x, y = np.meshgrid(x, y)
    occupancy = np.array([])
    l = np.array([(0,1), (1,2), (2,3), (3,0)])
    
    for i in l:
        pointa = corners[i[0]]
        pointb = corners[i[1]]
        x = pointb[0] - pointa[0]
        y = pointb[1] - pointa[1]
        
        if x == 0 and y ==0:
            print("Waring: points might not be in the correct order")

        elif x == 0: # if poth of the points are in y axis
            y = np.linspace(pointa[1], pointb[1], abs(y)+1)
            x = np.zeros_like(y) + pointa[0]
            occupancy = np.append(occupancy, np.array([x, y]).T)
        
        elif y == 0: # if both of the points are in x axis 
            x = np.linspace(pointa[0], pointb[0], abs(x)+1)
            y = np.zeros_like(x)
            occupancy = np.append(occupancy, np.array([x, y]).T)

        else: # if the points are not in the same axis
            m = y/x
            b = pointa[1] - m*pointa[0]
            x = np.linspace(pointa[0], pointb[0], max(pointa[0], pointb[0]))
            y = m*x + b
            occupancy = np.append(occupancy, np.floor(np.array([x, y]).T))

    occupancy = np.array(occupancy).reshape(-1,2).astype(int)
    grid[occupancy[:,0], occupancy[:,1]] = 1
    return grid

def get_obstacle_ocupancy(radius):
    x = np.arange(0, radius*2+1, 1)
    y = np.arange(0, radius*2+1, 1)
    X, Y = np.meshgrid(x, y)
    distances = np.sqrt((X - radius)**2 + (Y - radius)**2)
    binary_map = distances <= radius
    binary_map = binary_map.astype(int)
    occupancy_xy = np.array(np.where(binary_map == 1)).T
    return occupancy_xy

def add_padding(grid, padding):
    """makes grids adjacent to the occupied grid to be occupied based on the padding value"""
    occupanct_grids_xy = np.array(np.where(grid == 1)).astype(np.int16)
    padding = int(np.floor(padding).astype(np.int16))

    obstacle_occupancy = get_obstacle_ocupancy(padding) - padding//2

    for point in occupanct_grids_xy.T:
        x = obstacle_occupancy + point
        for p in x:
            try:
                grid[p[0], p[1]] = 1
            except IndexError:
                pass

    # step = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
    # for p in range(padding):
    #     for i in range(occupanct_grids_xy.shape[1]):
    #         for s in step:
    #             try:
    #                 if grid[occupanct_grids_xy[0, i] + s[0]*p, occupanct_grids_xy[1, i] + s[1]*p] == 0:
    #                     grid[occupanct_grids_xy[0, i] + s[0]*p, occupanct_grids_xy[1, i] + s[1]*p] = 1
    #             except IndexError:
    #                 pass
    
    return grid

def plan(current_state, waypoint, left_boundary, right_boundary, obstacles_centers):
    global plot_shown
    current_position = np.array(current_state[0])
    waypoint = np.array(waypoint)
    left_boundary = np.array(left_boundary)
    right_boundary = np.array(right_boundary)

    scale = 2
    padding = 3 # padding around the occupancy grid, in meters (unit of measurement value)
    obstacle_radius = 2.5 # radius of the obstacle, in meters (unit of measurement value)

    # transform to make current position the origin
    left_lane_t = left_boundary - current_position
    right_lane_t = right_boundary - current_position 
    current_position_t = current_position - current_position
    waypoint_t  =  waypoint - current_position 
    if obstacles_centers is not None: obstacles_centers = obstacles_centers -  current_position
    
    # Getting the transform to shift all the values to positive values as grid index cannot have negative values
    min_transform = get_transform_min(left_lane_t, right_lane_t, current_position_t, waypoint_t, obstacles_centers, obstacle_radius)
    
    left_lane_t = left_lane_t - min_transform
    right_lane_t = right_lane_t - min_transform
    current_position_t = current_position_t - min_transform
    waypoint_t = waypoint_t - min_transform
    if obstacles_centers is not None: obstacles_centers = obstacles_centers - min_transform

    # tvec container all the transform without roation values untill now, we will use this to regain our actual values
    tvec = min_transform + current_position

    # scaling the grid to make it smaller or bigger depending the resolution we want, 
    # lower the better for calculatioins, but too low will lead to no path solution
    grid_left_lane_points  = scaled_points_on_grid(left_lane_t, scale)
    grid_right_lane_points = scaled_points_on_grid(right_lane_t, scale)
    grid_waypoint          = scaled_points_on_grid(waypoint_t, scale)
    grid_current_position  = scaled_points_on_grid(current_position_t, scale)
    if obstacles_centers is not None:  grid_obstacles_centers = scaled_points_on_grid(obstacles_centers, scale)
    

    # Since all the values are positive due to previous transformations, 
    # we can use the max values to get the size of the grid.
    # This allows us to make dynamic grid size based on the obsticles, goal, and start positions
    
    max = get_transform_max(left_lane_t, right_lane_t, current_position_t, waypoint_t, obstacles_centers, obstacle_radius)

    # creating the grid and populating it with obsticles
    grid_shape = (np.ceil(np.array(max)*scale)).astype(int)
    grid = np.zeros(grid_shape)
    grid[grid_left_lane_points[:, 0], grid_left_lane_points[:, 1]] = 1
    grid[grid_right_lane_points[:, 0], grid_right_lane_points[:, 1]] = 1
    grid[grid_waypoint[0], grid_waypoint[1]] = 6 # any value other than 1 is considered not an obsticle, this is purely for visualization
    grid[grid_current_position[0], grid_current_position[1]] = 2 # same as above

    grid  = add_padding(grid, padding*scale)

    obstacle_occupancies = None
    if obstacles_centers is not None: 

        obstacle_occupancy = get_obstacle_ocupancy(obstacle_radius*scale)
        obstacle_occupancies = np.array([obstacle_occupancy + obstacle_center for obstacle_center in grid_obstacles_centers])

        for single_obsticle_occupancy in obstacle_occupancies:
            for point in single_obsticle_occupancy:
                if point[0] >= 0 and point[1] >= 0 and point[0] < grid_shape[0] and point[1] < grid_shape[1]:
                    grid[point[0], point[1]] = 1

      
    
    # print(grid)

    path, cost = a_star(grid, heuristic, 
                            (grid_current_position[0], grid_current_position[1]) , 
                            (grid_waypoint[0], grid_waypoint[1]))
    if path is not None:
        path = np.array(path)
        pruned_path = prune_path(path)

        #transforming path to the original coordinate system
        pruned_path_t = (pruned_path / scale) + min_transform + current_position
        plot(pruned_path, grid, grid_current_position, grid_waypoint, obstacle_occupancies,  tvec, scale)
        
        return pruned_path_t[1]

    else: 
        return None



class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """


    LEFT      = (-1, 0, 1)
    RIGHT     = (1, 0, 1)
    UP        = (0, 1, 1)
    DOWN      = (0, -1, 1)
    UP_LEFT   = (-1, 1, 1.41421)
    UP_RIGHT  = (1, 1, 1.41421)
    DOWN_LEFT = (-1, -1, 1.41421)
    DOWN_RIGHT= (1, -1, 1.41421)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])

    def __str__(self):
        return str(self.name)

def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    if x -1 < 0: valid_actions.remove(Action.LEFT)
    if x + 1 > n: valid_actions.remove(Action.RIGHT)
        
    if y - 1 < 0: valid_actions.remove(Action.DOWN)
    if y + 1 > m: valid_actions.remove(Action.UP)

    if x - 1 < 0 or y - 1 < 0: valid_actions.remove(Action.DOWN_LEFT)
    if x - 1 < 0 or y + 1 > m: valid_actions.remove(Action.UP_LEFT)
    if x + 1 > n or y - 1 < 0: valid_actions.remove(Action.DOWN_RIGHT)
    if x + 1 > n or y + 1 > m: valid_actions.remove(Action.UP_RIGHT)

    
    for actions in valid_actions:
        if grid[x + actions.delta[0], y + actions.delta[1]] == 1:
            # for act in valid_actions:
            #     valid_actions.remove(act)

            valid_actions.remove(actions)

    return valid_actions

def a_star(grid, h, start, goal):
    """
    Given a grid and heuristic function returns
    the lowest cost path from start to goal.
    """

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]
        
        
        if current_node == goal:
            # print('Found a path.')
            found = True
            break
        else:
            # Get the new vertexes connected to the current vertex
            for a in valid_actions(grid, current_node):
                next_node = (current_node[0] + a.delta[0], current_node[1] + a.delta[1])
                new_cost = current_cost + a.cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node, a)

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
        return None, None

    return path[::-1], path_cost

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


class Agent():
    
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        # self.decisionModule = VehicleDecision()
        self.prev_error_yaw = 0
        self.prev_error_vel = 0
        plt.ion()
        plt.show()

    def execute(self, currentPose, targetPose):
        """
            This function takes the current state of the vehicle and
            the target state to compute low-level control input to the vehicle
            Inputs:
                currentPose: ModelState, the current state of vehicle
                targetPose: The desired state of the vehicle
        """

        #currentEuler = currentPose[1]
        curr_x = currentPose[0][0]
        curr_y = currentPose[0][1]
        curr_theta = currentPose[1][1]
        curr_theta = (curr_theta+360)%360
        
        target_x = targetPose[0]
        target_y = targetPose[1]
        target_v = targetPose[2]
        target_theta =  np.arctan2(target_y-curr_y, target_x-curr_x)*180/np.pi
        target_theta = (target_theta+360)%360

        k_s = 0.1
        k_ds = 1
        k_n = 0.038
        k_theta = 0.25
        decelerate_when_steering = 15

        # compute errors
        dx = target_x - curr_x
        dy = target_y - curr_y

        xError = (dx) * np.cos(currentPose[1][1]) + (dy) * np.sin(currentPose[1][1])
        yError = -(dx) * np.sin(currentPose[1][1]) + (dy) * np.cos(currentPose[1][1])
        
        curr_v = np.sqrt(currentPose[2][0]**2 + currentPose[2][1]**2)

        thetaError = target_theta - curr_theta
        #print(curr_theta, target_theta, thetaError)

        p = (k_n * yError)
        d = (k_theta * thetaError)
        delta = p + d


        #target_v = max(0,target_v - decelerate_when_steering*abs(delta))
        #print(delta, target_v)


        vError = target_v - curr_v


        # error_vel = target_v - curr_v
        # vp = 0.1
        # vd = 1
        # throttle = vp * error_vel + vd * (error_vel - self.prev_error_vel)
        # throttle = max(-1, min(1, throttle))
        # self.prev_error_vel = error_vel

        delta = k_n * yError
        print(f"delta: {delta}")
        delta = max(-1, min(1, delta))
        #delta = (delta + 1) / 2
        #delmin = 
        # Checking if the vehicle need to stop
        if target_v >= 0:
            v = xError * k_s + vError * k_ds
            print("v:", v)
            if vError < 0:
                print("vError:", vError)
                
                throttle = 0
                brake = 0.5
                print("=====Deccelerating======")

            else:
                if(curr_v >= 15):
                    throttle = 0
                    brake = 1.0
                else:
                    throttle = v * 0.1
                    brake = 0
                print("************Execcuting Go Condition************")
                print(f"throttle: {throttle}, delta: {delta}")
            return [throttle,delta, brake]
        #
            #Send computed control input to vehicle
        #if target_v >= 0:
        
        #v = max(0, min(1, v))


        #print(f"dela {delta}, d: {d}, theta_error: {thetaError}")
        #return [throttle,delta] # [speed, steering_angle]

        # else:
        #     print("************Execcuting Stop Condition************")
        #     return self.stop()

    def get_throttle_and_steering(self, currentPose, targetPose):
        
        current_yaw = currentPose[1][1]
        current_x = currentPose[0][0]
        current_y = currentPose[0][1]
        currernt_v = np.sqrt(currentPose[2][0]**2 + currentPose[2][1]**2)
        
        target_vel = targetPose[2]
        target_x = targetPose[0]
        target_y = targetPose[1]
        #print("current_x, current_y, target_x, target_y", current_x, current_y, target_x, target_y)


        target_yaw = np.rad2deg(np.arctan2(target_y-current_y,target_x-current_x))
        #print(target_yaw)
        #print("target_yaw before", target_yaw)
        error_yaw =  target_yaw - current_yaw # this one was working 
        # error_yaw =  current_yaw - target_yaw
        
        if (error_yaw < -90.0 or error_yaw > 90.0):

            #error_yaw = error_yaw % 90
            if (error_yaw > 180):
                error_yaw = 360 - error_yaw
                error_yaw = 180 - error_yaw
                error_yaw = -error_yaw
            elif (error_yaw < -180):
                error_yaw = 360 + error_yaw
                error_yaw = 180 - error_yaw
            elif (error_yaw > 90):
                error_yaw = 180 - error_yaw
            elif (error_yaw < -90):
                error_yaw = 180 + error_yaw
                error_yaw = -error_yaw
                
            else:
                error_yaw = error_yaw % 90
            print("Corrected error_yaw")

        # if target_yaw >= 0 and current_yaw < 0:
        #     error_yaw = min(target_yaw - current_yaw, target_yaw - (current_yaw+360))
        # elif target_yaw < 0 and current_yaw >= 0:
        #     error_yaw = min(target_yaw - current_yaw, (target_yaw + 360) - current_yaw)
        # else:
        #     error_yaw = target_yaw - current_yaw

        error_vel = target_vel - currernt_v
        # print("error_yaw", error_yaw)

        sp = 0.01
        sd = 0.0
        steering = sp * error_yaw + sd * (error_yaw - self.prev_error_yaw)
        print("steering, current_yaw, target_yaw, error_yaw", steering, current_yaw, target_yaw, error_yaw)

    # --------------------------------------------
        vp = 0.03
        vd = 0
        throttle = vp * error_vel + vd * (error_vel - self.prev_error_vel)
        
        self.prev_error_yaw = error_yaw
        self.prev_error_vel = error_vel

        throttle = max(0, min(1, throttle))
        steering = max(-1, min(1, steering))
        #print("steering after:", steering) 
        # print("throttle, steering", throttle, steering, target_vel)
        
        return throttle, steering

    def run_step(self, filtered_obstacles, waypoints, vel, transform, boundary):
        """
        Execute one step of navigation.
        Args:
        filtered_obstacles
            - Type:        List[carla.Actor(), ...]
            - Description: All actors except for EGO within sensoring distance
        waypoints 
            - Type:         List[[x,y,z], ...] 
            - Description:  List All future waypoints to reach in (x,y,z) format
        vel
            - Type:         carla.Vector3D 
            - Description:  Ego's current velocity in (x, y, z) in m/s
        transform
            - Type:         carla.Transform 
            - Description:  Ego's current transform
        boundary 
            - Type:         List[List[left_boundry], List[right_boundry]]
            - Description:  left/right boundary each consists of 20 waypoints,
                            they defines the track boundary of the next 20 meters.
        Return: carla.VehicleControl()
        """
        global flag
        if (flag == 0):
            print("List of waypoints", waypoints)
            flag = 1
        # Actions to take during each simulation step
        # Feel Free to use carla API; however, since we already provide info to you, using API will only add to your delay time
        # Currently the timeout is set to 10s
        # currState: [Loaction, Rotation, Velocity]
        currPose = [[transform.location.x,transform.location.y], 
                     [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
                      [vel.x, vel.y, vel.z]]

        if len(filtered_obstacles) > 0: 
            obstacleList = [obj.get_location() for obj in filtered_obstacles]
            obstacleList = [[obj.x, obj.y] for obj in obstacleList]
        else:
            obstacleList = None

        boundary_left = [[element.transform.location.x, element.transform.location.y] for element in boundary[0]]
        boundary_right = [[element.transform.location.x, element.transform.location.y] for element in boundary[1]]
        waypoint = [waypoints[0][0], waypoints[0][1]]
        # print("boundary_lane_markers:", waypoint)


        # # refState = self.decisionModule.get_ref_state(currState, obstacleList,
        #                                               waypoint, boundary_lane_markers)
        target_speed = 15
        resp = plan(currPose, 
                        waypoint,
                        boundary_left, 
                        boundary_right, 
                        obstacleList) # [x, y, v]

        if resp is not None:
            target_x, target_y = resp[0], resp[1]
            target_speed = 15
        else:
            target_x, target_y = waypoint[0], waypoint[1]
            target_speed = 15

        target_state = [target_x, target_y, target_speed]
        waypoint_state = [waypoint[0], waypoint[1], target_speed]


        throttle, steer = self.get_throttle_and_steering(currPose, target_state)
        # throttle, steer, brake = self.execute(currPose, target_state)
        # given target state, compute control input


        # speed, angle = self.get_angle_and_speed(currState, target_state)
        

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        # control.brake = brake
        return control