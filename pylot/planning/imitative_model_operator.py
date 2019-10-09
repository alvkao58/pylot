from collections import deque
import numpy as np
import threading
import tensorflow as tf

import carla

from erdos.message import WatermarkMessage
from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging

import pylot.utils
from pylot.map.hd_map import HDMap
from pylot.planning.utils import get_distance
from pylot.simulation.carla_utils import get_map, to_carla_location
from pylot.simulation.utils import Location, Rotation, Transform
from pylot.perception.messages import ObjTrajectory, ObjTrajectoriesMessage

from madim.madim import models, carla_json_loader
WAYPOINT_COMPLETION_THRESHOLD = 0.9

class ImitativeModelOperator(Op):
    """ Wrapper operator for the imitative models."""

    def __init__(self,
                 name,
                 flags,
                 goal_location=None,
                 log_file_name=None,
                 csv_file_name=None):
        super(ImitativeModelOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._flags = flags
#        self._planning_graph = tf.Graph() 
#         # Load the model from the model file.
#         with self._planning_graph.as_default():
#             graph_def = tf.GraphDef()
#             with tf.gfile.GFile(self._flags.imitative_planning_model_path, 'rb') as fid:
#                 serialized_graph = fid.read()
#                 graph_def.ParseFromString(serialized_graph)
#                 tf.import_graph_def(graph_def, name='')
# 
#         self._gpu_options = tf.GPUOptions(
#             per_process_gpu_memory_fraction=flags.planning_imitative_gpu_memory_fraction)
#         # Create a TensorFlow session.
#         self._tf_session = tf.Session(
#             graph=self._planning_graph,
#             config=tf.ConfigProto(gpu_options=self._gpu_options))
#         # Get the tensors we're interested in.
#         # TODO: Figure these out from the pre-trained models.

        print ("TRYING MODEL...........................")
        self._r2p2_model = models.R2P2_RNN(None, device='cpu:0')
        # Try to load model.
#self._r2p2_model.load_weights('/home/erdos/workspace/pylot/dependencies/madim/test_model')
        print ("DONE IMPORTING")
        # Queues of incoming data.
        self._can_bus_msgs = deque()
        self._lidar = deque()
        self._ego_vehicle_id = None
        self._trajectories = deque()
        self._lock = threading.Lock()
        self._frame_cnt = 0
        # There is no track flag present, i.e. we're not running in challenge mode.
        # Thus, we can directly get the map from the simulator.
        if not hasattr(self._flags, 'track'):
            self._map = HDMap(get_map(self._flags.carla_host,
                                      self._flags.carla_port,
                                      self._flags.carla_timeout),
                              log_file_name)
            # TODO: Fix
            assert goal_location, 'Planner has not received a goal location'
            # Transform goal location to carla.Location
            self._goal_location = carla.Location(*goal_location)
        else:
            # In challenge mode, we don't have access to locations of other vehicles.
            raise NotImplementedError

    @staticmethod
    def setup_streams(input_streams, output_stream_name):
        input_streams.filter(
            pylot.utils.is_lidar_stream).add_callback(
                ImitativeModelOperator.on_lidar_update)
        # Past+present location of ego-vehicle (and other vehicles)
        input_streams.filter(
            pylot.utils.is_tracking_stream).add_callback(
                ImitativeModelOperator.on_trajectory_update)
        input_streams.filter(
            pylot.utils.is_ground_vehicle_id_stream).add_callback(
                ImitativeModelOperator.on_ground_vehicle_id_update)
        input_streams.filter(pylot.utils.is_can_bus_stream).add_callback(
            ImitativeModelOperator.on_can_bus_update)
        input_streams.add_completion_callback(
            ImitativeModelOperator.on_notification)
        # Outputs fine-grained waypoints.
        print ("CREATING OUTPUT STREAM", output_stream_name)
        return [pylot.utils.create_prediction_stream(output_stream_name)]

    def synchronize_msg_buffers(self, timestamp, buffers):
        for buffer in buffers:
            while (len(buffer) > 0 and buffer[0].timestamp < timestamp):
                buffer.popleft()
            if len(buffer) == 0:
                return False
            assert buffer[0].timestamp == timestamp
        return True

    def __update_waypoints(self, ego_location):
        """ Updates the waypoints.

        Depending on setup, the method either recomputes the waypoints
        between the ego vehicle and the goal location, or just garbage collects
        waypoints that have already been achieved.

        """
        self._waypoints = self._map.compute_waypoints(
            to_carla_location(ego_location),
            self._goal_location)
        self.__remove_completed_waypoints(ego_location)
        if not self._waypoints or len(self._waypoints) == 0:
            # If waypoints are empty (e.g., reached destination), set waypoint
            # to current vehicle location.
            self._waypoints = deque([self._vehicle_transform])

#print ("Closest waypoint", len(self._waypoints), self._waypoints[0].location, ego_location)
        return (self._waypoints)

    def __remove_completed_waypoints(self, ego_location):
        """ Removes waypoints that the ego vehicle has already completed.

        The method first finds the closest waypoint, removes all waypoints
        that are before the closest waypoint, and finally removes the
        closest waypoint if the ego vehicle is very close to it
        (i.e., close to completion)."""
        min_dist = 10000000
        min_index = 0
        index = 0
        for waypoint in self._waypoints:
            # XXX(ionel): We only check the first 10 waypoints.
            if index > 10:
                break
            dist = get_distance(waypoint.location,
                                ego_location)
            if dist < min_dist:
                min_dist = dist
                min_index = index

        # Remove waypoints that are before the closest waypoint. The ego
        # vehicle already completed them.
        while min_index > 0:
            self._waypoints.popleft()
            min_index -= 1

        # The closest waypoint is almost complete, remove it.
        if min_dist < WAYPOINT_COMPLETION_THRESHOLD:
            self._waypoints.popleft()

    def _get_rectifying_player_transform(self, can_bus):
        """ Build transform to correct for pitch or roll of car. """
        ptx = can_bus.transform
        p_rectify_pos = Location(0, 0, 0)
        p_rectify_rot = Rotation(-ptx.rotation.pitch, 0, -ptx.rotation.roll)
        return Transform(p_rectify_pos, p_rectify_rot)

    def _get_rectified_sensor_transform(self, lidar_transform, can_bus):
        p_rectify = self._get_rectifying_player_transform(can_bus)
        
        # Transform to car frame, then undo the pitch and roll of the car frame.
        return p_rectify * lidar_transform 

    def _get_occupancy_grid(self, lidar_transform, lidar_point_cloud, can_bus, lidar_params):
        """Get occupancy grid(s) (indicators of occupancy) at various heights."""

        # Get world -> rectified lidar transform
        lidar_transform = self._get_rectified_sensor_transform(lidar_transform, can_bus)

        # Transform points to the car frame
        lidar_points_at_car = np.asarray(lidar_transform.transform_points(lidar_point_cloud))

        permute = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]], dtype=np.float32)
#lidar_points_at_car = (permute @ lidar_points_at_car.T).T
        lidar_points_at_car = (np.dot(permute, lidar_points_at_car.T)).T
        z_threshold = -4.5  # sees low objects, but sometimes the lidar rolls and sees the road as an obstacle
        # z_threshold = -3.0  # doesn't see low objects, but the lidar is fine

        above_mask = lidar_points_at_car[:, 2] > z_threshold

        def get_occupancy_from_masked_lidar(mask):
            masked_lidar = lidar_points_at_car[mask]
            meters_max = lidar_params['meters_max']
            pixels_per_meter = lidar_params['pixels_per_meter']
            xbins = np.linspace(-meters_max, meters_max, meters_max * 2 * pixels_per_meter + 1)
            ybins = xbins
            grid = np.histogramdd(masked_lidar[..., :2], bins=(xbins, ybins))[0]
            grid[grid > 0.] = lidar_params['val_obstacle']
            return grid

        feats = ()
        feats += (get_occupancy_from_masked_lidar(above_mask),)
        feats += (get_occupancy_from_masked_lidar((1 - above_mask).astype(np.bool)),)
        return np.stack(feats, axis=-1)

    def on_notification(self, msg):
        # Pop the oldest message from each buffer.
        with self._lock:
            if not self._ego_vehicle_id:
                return
            if not self.synchronize_msg_buffers(
                    msg.timestamp,
                    [self._lidar, self._trajectories, self._can_bus_msgs]):
                return
            lidar_msg = self._lidar.popleft()
            trajectories_msg = self._trajectories.popleft()
            can_bus_msg = self._can_bus_msgs.popleft()

        self._logger.info('Timestamps {} {} {}'.format(
            lidar_msg.timestamp, trajectories_msg.timestamp,
            can_bus_msg.timestamp))

        assert (lidar_msg.timestamp == trajectories_msg.timestamp ==
                can_bus_msg.timestamp)

        self._frame_cnt += 1
        ego_vehicle_trajectory = None
        for agent in trajectories_msg.obj_trajectories:
            if agent.obj_id != self._ego_vehicle_id:
                continue
            ego_vehicle_trajectory = agent.trajectory
        assert (ego_vehicle_trajectory is not None)
        # Process lidar using splat_lidar
        lidar_params = {'meters_max': 50,
                        'pixels_per_meter': 2,
                        'val_obstacle': 1.0}
        grid = self._get_occupancy_grid(lidar_msg.transform, lidar_msg.point_cloud, can_bus_msg.data, lidar_params)
        processed_trajectory = self.process_trajectory(ego_vehicle_trajectory)
        
        if processed_trajectory.shape[1] >= 2:
            predictions = self._r2p2_model.predict(processed_trajectory, np.expand_dims(grid, axis=0).astype(np.float32))
        else:
            # We only have one observation so far (which is not enough for
            # R2P2), so we predict that we continue to stay in the same
            #location.
            # TODO (alvin): flag for number of observations
            predictions = np.zeros((1, 10, 2))
            #predictions[0, :] = processed_trajectory[0]
        self.__update_waypoints(can_bus_msg.data.transform.location)
        # Send to output stream.
        predicted_ego_vehicle = ObjTrajectory('vehicle', self._ego_vehicle_id, np.array(predictions[0]))
        output_msg = ObjTrajectoriesMessage([predicted_ego_vehicle], msg.timestamp) 
        self.get_output_stream('prediction').send(output_msg)
        self.get_output_stream('prediction')\
            .send(WatermarkMessage(msg.timestamp))

    def process_trajectory(self, trajectory):
        processed_trajectory = np.zeros((1, len(trajectory), 2))
        for i in range(len(trajectory)):
            loc = trajectory[i]
            processed_trajectory[0][i][0] = loc.x
            processed_trajectory[0][i][1] = loc.y
        return processed_trajectory 

    def on_can_bus_update(self, msg):
        with self._lock:
           self._can_bus_msgs.append(msg)

    def on_ground_vehicle_id_update(self, msg):
        with self._lock:
            self._ego_vehicle_id = msg.data

    def on_lidar_update(self, msg):
        with self._lock:
            self._lidar.append(msg)

    def on_trajectory_update(self, msg):
        with self._lock:
            self._trajectories.append(msg)

