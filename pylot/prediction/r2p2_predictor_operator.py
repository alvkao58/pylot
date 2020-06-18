import math
import time
from collections import deque

import erdos

import numpy as np

import torch

try:
    from pylot.prediction.prediction.r2p2.r2p2_model import R2P2
except ImportError:
    raise Exception('Error importing R2P2.')

import pylot.prediction.flags
from pylot.prediction.messages import PredictionMessage
from pylot.prediction.obstacle_prediction import ObstaclePrediction
import pylot.prediction.utils
from pylot.utils import Location, Transform, time_epoch_ms


class R2P2PredictorOperator(erdos.Operator):
    """Wrapper operator for R2P2 ego-vehicle prediction module.

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
        lidar_setup (:py:class:`pylot.drivers.sensor_setup.LidarSetup`): Setup
            of the lidar. This setup is used to get the maximum range of the
            lidar.
    """
    def __init__(self, point_cloud_stream, tracking_stream, prediction_stream,
                 flags, lidar_setup):
        print("WARNING: R2P2 predicts only vehicle trajectories")
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            self.config.name + '-csv', self.config.csv_log_file_name)
        self._flags = flags

        self._device = torch.device('cuda')
        self._r2p2_model = R2P2().to(self._device)
        state_dict = torch.load(flags.r2p2_model_path)
        self._r2p2_model.load_state_dict(state_dict)

        point_cloud_stream.add_callback(self.on_point_cloud_update)
        tracking_stream.add_callback(self.on_trajectory_update)
        erdos.add_watermark_callback([point_cloud_stream, tracking_stream],
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

        all_vehicles = [
            obstacle_trajectory
            for obstacle_trajectory in tracking_msg.obstacle_trajectories
            if obstacle_trajectory.obstacle.is_vehicle()
        ]
        closest_vehicles, ego_vehicle = \
            pylot.prediction.utils.get_closest_vehicles(
                all_vehicles,
                self._flags.prediction_radius,
                self._flags.prediction_ego_agent)
        num_predictions = len(closest_vehicles)
        self._logger.info('@{}: Getting predictions for {} vehicles'.format(
            timestamp, num_predictions))

        start_time = time.time()
        if num_predictions == 0:
            self._logger.debug(
                '@{}: no vehicles to make predictions for'.format(timestamp))
            prediction_stream.send(PredictionMessage(timestamp, []))
            return

        closest_vehicles_ego_transforms, closest_trajectories, binned_lidars = \
            self._preprocess_input(closest_vehicles, ego_vehicle,
                                   point_cloud_msg.point_cloud.points)

        # Run the forward pass.
        z = torch.tensor(
            np.random.normal(size=(num_predictions,
                                   self._flags.prediction_num_future_steps,
                                   2))).to(torch.float32).to(self._device)
        closest_trajectories = torch.tensor(closest_trajectories).to(
            torch.float32).to(self._device)
        binned_lidars = torch.tensor(binned_lidars).to(torch.float32).to(
            self._device)
        model_start_time = time.time()
        prediction_array, _ = self._r2p2_model.forward(z, closest_trajectories,
                                                       binned_lidars)
        model_runtime = (time.time() - model_start_time) * 1000
        self._csv_logger.debug("{},{},{},{:.4f}".format(
            time_epoch_ms(), timestamp.coordinates[0],
            'r2p2-modelonly-runtime', model_runtime))
        prediction_array = prediction_array.cpu().detach().numpy()

        obstacle_predictions_list = self._postprocess_predictions(
            prediction_array, closest_vehicles,
            closest_vehicles_ego_transforms)
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.debug("{},{},{},{:.4f}".format(
            time_epoch_ms(), timestamp.coordinates[0], 'r2p2-runtime',
            runtime))
        prediction_stream.send(
            PredictionMessage(timestamp, obstacle_predictions_list))

    def _preprocess_input(self, closest_vehicles, ego_vehicle, point_cloud):
        num_predictions = len(closest_vehicles)

        closest_vehicles_ego_transforms = \
            pylot.prediction.utils.get_closest_vehicles_ego_transforms(
                closest_vehicles, ego_vehicle)

        # Rotate and pad the closest trajectories.
        closest_trajectories = []
        for i in range(num_predictions):
            cur_trajectory = np.stack([[point.location.x,
                                        point.location.y,
                                        point.location.z] \
                for point in closest_vehicles[i].trajectory])

            # Remove z-coordinate from trajectory.
            closest_trajectories.append(
                closest_vehicles_ego_transforms[i].inverse_transform_points(
                    cur_trajectory)[:, :2])
        closest_trajectories = np.stack(
            [pylot.prediction.utils.pad_trajectory(t, self._flags.prediction_num_past_steps) \
                for t in closest_trajectories])

        # For each vehicle, transform the lidar point cloud to that vehicle's
        # coordinate frame for purposes of prediction.
        binned_lidars = []
        for i in range(num_predictions):
            rotated_point_cloud = closest_vehicles_ego_transforms[
                i].inverse_transform_points(point_cloud)
            binned_lidars.append(
                pylot.prediction.utils.get_occupancy_grid(rotated_point_cloud,
                    self._lidar_setup.transform.location.z,
                    int(self._lidar_setup.get_range_in_meters())))
        binned_lidars = np.concatenate(binned_lidars)

        return closest_vehicles_ego_transforms, closest_trajectories, binned_lidars
        
    def _postprocess_predictions(self, prediction_array, vehicles,
                                 vehicles_ego_transforms):
        # Transform each predicted trajectory to be in relation to the
        # ego-vehicle, then convert into an ObstaclePrediction. Because R2P2
        # performs top-down prediction, we assume the vehicle stays at the same
        # height as its last location.
        obstacle_predictions_list = []
        num_predictions = len(vehicles_ego_transforms)

        for idx in range(num_predictions):
            cur_prediction = prediction_array[idx]

            last_location = vehicles_ego_transforms[idx].location
            predictions = []
            for t in range(self._flags.prediction_num_future_steps):
                cur_point = vehicles_ego_transforms[idx].transform_points(
                    np.array([[
                        cur_prediction[t][0], cur_prediction[t][1],
                        last_location.z
                    ]]))[0]
                # R2P2 does not predict vehicle orientation, so we use our
                # estimated orientation of the vehicle at its latest location.
                predictions.append(
                    Transform(location=Location(cur_point[0], cur_point[1],
                                                cur_point[2]),
                              rotation=vehicles_ego_transforms[idx].rotation))

            # Probability; currently a filler value because we are taking
            # just one sample from distribution
            obstacle_predictions_list.append(
                ObstaclePrediction(vehicles[idx], vehicles_ego_transforms[idx],
                                   1.0, predictions))
        return obstacle_predictions_list

    def on_point_cloud_update(self, msg):
        self._logger.debug('@{}: received point cloud message'.format(
            msg.timestamp))
        self._point_cloud_msgs.append(msg)

    def on_trajectory_update(self, msg):
        self._logger.debug('@{}: received trajectories message'.format(
            msg.timestamp))
        self._tracking_msgs.append(msg)
