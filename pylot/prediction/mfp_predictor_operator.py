import erdos

from typing import List, Set, Dict, Tuple, Optional, Union, Any
from collections import deque
import argparse
import gin
import math
import numpy as np
import os
from pylot.utils import time_epoch_ms
import time
import torch

from multiple_futures_prediction.model_ngsim import mfpNet
from multiple_futures_prediction.train_ngsim import Params

import pylot.prediction.flags
from pylot.prediction.messages import PredictionMessage
from pylot.prediction.obstacle_prediction import ObstaclePrediction
import pylot.prediction.utils
from pylot.utils import Location, Rotation, Transform, Vector2D

class MFPPredictorOperator(erdos.Operator):
    """Wrapper operator for MFP prediction module.

    Args:
        point_cloud_stream (:py:class:`erdos.ReadStream`, optional): Stream on
            which point cloud messages are received.
        tracking_stream (:py:class:`erdos.ReadStream`):
            Stream on which
            :py:class:`~pylot.perception.messages.ObstacleTrajectoriesMessage`
            are received.
        prediction_stream (:py:class:`erdos.ReadStream`): Stream on which
            :py:class:`~pylot.prediction.messages.PredictionMessage`
            messages are published.
        lidar_setup (:py:class:`pylot.drivers.sensor_setup.LidarSetup`): Lidar
            setup is used to get the maximum range of the lidar.
        """

    def __init__(self,
                 point_cloud_stream,
                 tracking_stream,
                 prediction_stream,
                 flags,
                 lidar_setup):
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._flags = flags

        self._device = torch.device('cuda')
        gin.parse_config_file('/home/erdos/workspace/pylot/dependencies/multiple_futures_prediction/configs/mfp2_ngsim.gin')
        params = Params(fut_len_orig_hz=self._flags.prediction_num_future_steps)()
        assert params.fut_len_orig_hz == self._flags.prediction_num_future_steps
        self._mfp_model = mfpNet(params).to(self._device)
        state_dict = torch.load(flags.mfp_model_path)
        self._mfp_model.load_state_dict(state_dict)

        point_cloud_stream.add_callback(self.on_point_cloud_update)
        tracking_stream.add_callback(self.on_trajectory_update)
        erdos.add_watermark_callback(
            [point_cloud_stream, tracking_stream],
            [prediction_stream], self.on_watermark)

        self._lidar_setup = lidar_setup

        self._point_cloud_msgs = deque()
        self._tracking_msgs = deque()

    @staticmethod
    def connect(point_cloud_stream, tracking_stream):
        prediction_stream = erdos.WriteStream()
        return [prediction_stream]

    def on_watermark(self, timestamp, prediction_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))

        point_cloud_msg = self._point_cloud_msgs.popleft()
        tracking_msg = self._tracking_msgs.popleft()

        start_time = time.time()
        nearby_vehicle_trajectories, nearby_trajectories_tensor, nearby_vehicle_ego_transforms, \
            binned_lidars_tensor, nbrs, nbrs_info = \
            self._preprocess_input(tracking_msg, point_cloud_msg)

        num_predictions = len(nearby_trajectories_tensor)
        self._logger.info(
            '@{}: Getting MFP predictions for {} vehicles'.format(
                timestamp, num_predictions))

        if num_predictions == 0:
            prediction_stream.send(PredictionMessage(timestamp, []))
            return

        # Run the forward pass.
        model_start_time = time.time()
        prediction_array = self._mfp_model.forward_eval(
            nearby_trajectories_tensor, nbrs, binned_lidars_tensor, nbrs_info,
            num_traj=1)
        model_runtime = (time.time() - model_start_time) * 1000
        self._csv_logger.debug("{},{},{},{:.4f}".format(
            time_epoch_ms(), timestamp.coordinates[0],
            'mfp-modelonly-runtime', model_runtime))
        prediction_array = prediction_array[0].cpu().detach().numpy()

        # Predictions for each vehicle are in coordinates wrt that vehicle;
        # transform back to ego coordinates.
        obstacle_predictions_list = self._postprocess_predictions(
            prediction_array, nearby_vehicle_trajectories, \
            nearby_vehicle_ego_transforms)
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.debug("{},{},{},{:.4f}".format(
            time_epoch_ms(), timestamp.coordinates[0], 'mfp-runtime',
            runtime))
        prediction_stream.send(
            PredictionMessage(timestamp, obstacle_predictions_list))

    def _preprocess_input(self, tracking_msg, point_cloud_msg):
        nearby_vehicle_trajectories, nearby_vehicle_ego_transforms = \
            tracking_msg.get_nearby_obstacles_info(
                self._flags.prediction_radius,
                lambda t: t.obstacle.is_vehicle())
        point_cloud = point_cloud_msg.point_cloud.points
        num_nearby_vehicles = len(nearby_vehicle_trajectories)
        if num_nearby_vehicles == 0:
            return [], [], [], []

        # Remove the z-coordinate of the trajectory.
        nearby_trajectories_tensor = []  # Pytorch tensor for network input.

        for i in range(num_nearby_vehicles):
            cur_trajectory = nearby_vehicle_trajectories[
                i].get_last_n_transforms(self._flags.prediction_num_past_steps)
            cur_trajectory = np.stack([[point.location.x, point.location.y]
                 for point in cur_trajectory])

            nearby_trajectories_tensor.append(cur_trajectory)

        nearby_trajectories_tensor = np.stack(nearby_trajectories_tensor, axis=1)
        nearby_trajectories_tensor = torch.tensor(
            nearby_trajectories_tensor).to(torch.float32).to(self._device)

        # Bin the LIDAR.
        occupancy_grid = pylot.prediction.utils.get_occupancy_grid(
            point_cloud, self._lidar_setup.transform.location.z,
            int(self._lidar_setup.get_range_in_meters()))
        occupancy_grid = np.transpose(occupancy_grid, axes=(0,3,1,2))
        binned_lidars = np.tile(occupancy_grid, (num_nearby_vehicles,1,1,1))
        #print (binned_lidars.shape)
        binned_lidars_tensor = torch.tensor(binned_lidars).to(
            torch.float32).to(self._device)

        # Fill in neighbor trajectories for each vehicle.
        nbrs = torch.zeros((self._flags.prediction_num_past_steps,
                         num_nearby_vehicles*(num_nearby_vehicles-1), 2))
        cnt = 0
        for i in range(num_nearby_vehicles):
            for j in range(num_nearby_vehicles):
                if i != j:
                    nbrs[:,cnt,:] = nearby_trajectories_tensor[:,j,:]
                    cnt += 1
        nbrs = nbrs.to(self._device)

        # Nbrs_info dictionary.
        nbrs_info = {}
        for i in range(num_nearby_vehicles):
          nbrs_info[i] = [(j,) for j in range(num_nearby_vehicles) if i != j]
        nbrs_info = [nbrs_info]

        return nearby_vehicle_trajectories, nearby_trajectories_tensor, \
               nearby_vehicle_ego_transforms, binned_lidars_tensor, nbrs, \
               nbrs_info

    def _postprocess_predictions(self, prediction_array, vehicle_trajectories,
                                 vehicle_ego_transforms):
        # Model predictions are already rotated to ego-vehicle orientation,
        # just need to translate them.
        obstacle_predictions_list = []
        num_predictions = len(vehicle_trajectories)

        for idx in range(num_predictions):
            cur_prediction = prediction_array[:,idx,:]
            predictions = [] 
            # Because MFP only predicts (x,y) coordinates, we assume the
            # vehicle stays at the same height as its last location.
            vehicle_ego_location = Transform(location=vehicle_ego_transforms[idx].location, rotation=Rotation())
            for t in range(self._flags.prediction_num_future_steps):
                cur_point = vehicle_ego_location.transform_points(
                    np.array([[
                        cur_prediction[t][0], cur_prediction[t][1],
                        vehicle_ego_transforms[idx].location.z
                    ]]))[0]
                predictions.append(
                    Transform(location=Location(cur_point[0], cur_point[1],
                                                cur_point[2]),
                              rotation=vehicle_ego_transforms[idx].rotation))

            obstacle_transform = vehicle_trajectories[idx].obstacle.transform
            obstacle_predictions_list.append(
                ObstaclePrediction(vehicle_trajectories[idx],
                                   obstacle_transform, 1.0, predictions))
                
        return obstacle_predictions_list

    def on_point_cloud_update(self, msg):
        self._logger.debug('@{}: received point cloud message'.format(
            msg.timestamp))
        self._point_cloud_msgs.append(msg)

    def on_trajectory_update(self, msg):
        self._logger.debug('@{}: received trajectories message'.format(
            msg.timestamp))
        self._tracking_msgs.append(msg)
