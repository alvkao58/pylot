"""
An RRT* planning operator that runs the RRT* algorithm defined under
pylot/planning/rrt_star/rrt_star.py.

Planner steps:
1. Get ego vehicle information from can_bus stream
2. Compute the potential obstacles using the predictions from prediction stream
3. Compute the target waypoint to reach
4. Construct state_space, target_space, start_state and run RRT*
5. Construct waypoints message and output on waypoints stream
"""
import carla
import collections
from collections import deque
import erdos
import itertools

from pylot.map.hd_map import HDMap
from pylot.planning.messages import WaypointsMessage
from pylot.planning.rrt_star.rrt_star import apply_rrt_star
from pylot.planning.rrt_star.utils import start_target_to_space
from pylot.planning.utils import get_waypoint_vector_and_angle
from pylot.simulation.utils import get_map
from pylot.utils import Location, Rotation, Transform

DEFAULT_OBSTACLE_LENGTH = 3  # 3 meters from front to back
DEFAULT_OBSTACLE_WIDTH = 2  # 2 meters from side to side
DEFAULT_TARGET_LENGTH = 1  # 1.5 meters from front to back
DEFAULT_TARGET_WIDTH = 1  # 1 meters from side to side
DEFAULT_DISTANCE_THRESHOLD = 20  # 20 meters ahead of ego
DEFAULT_NUM_WAYPOINTS = 50  # 50 waypoints to plan for


class RRTStarPlanningOperator(erdos.Operator):
    """ RRTStar Planning operator for Carla 0.9.x."""
    def __init__(self,
                 can_bus_stream,
                 prediction_stream,
                 waypoints_stream,
                 name,
                 flags,
                 goal_location,
                 log_file_name=None,
                 csv_file_name=None):
        """
        Initialize the RRT* planner. Setup logger and map attributes.

        Args:
            name: Name of the operator.
            flags: Config flags.
            goal_location: Global goal carla.Location for planner to route to.
        """
        can_bus_stream.add_callback(self.on_can_bus_update)
        prediction_stream.add_callback(self.on_prediction_update)
        erdos.add_watermark_callback([can_bus_stream, prediction_stream],
                                     [waypoints_stream], self.on_watermark)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdos.utils.setup_csv_logging(
            name + '-csv', csv_file_name)
        self._flags = flags

        self._wp_index = 9
        self._waypoints = None
        self._carla_map = get_map(self._flags.carla_host,
                                  self._flags.carla_port,
                                  self._flags.carla_timeout)
        self._hd_map = HDMap(self._carla_map, log_file_name)
        self._goal_location = goal_location

        self._can_bus_msgs = deque()
        self._prediction_msgs = deque()

    @staticmethod
    def connect(can_bus_stream, prediction_stream):
        waypoints_stream = erdos.WriteStream()
        return [waypoints_stream]

    def on_can_bus_update(self, msg):
        self._logger.debug('@{}: received can bus message'.format(
            msg.timestamp))
        self._can_bus_msgs.append(msg)

    def on_prediction_update(self, msg):
        self._logger.debug('@{}: received prediction message'.format(
            msg.timestamp))
        self._prediction_msgs.append(msg)

    def on_watermark(self, timestamp, waypoints_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        # get ego info
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform

        # get obstacles
        obstacle_map = self._build_obstacle_map(vehicle_transform)

        # compute goals
        target_location = self._compute_target_location(vehicle_transform)

        # run rrt*
        path, cost = self._run_rrt_star(vehicle_transform, target_location,
                                        obstacle_map)

        # convert to waypoints if path found, else use default waypoints
        if cost is not None:
            path_transforms = []
            for point in path:
                p_loc = self._carla_map.get_waypoint(
                    carla.Location(x=point[0], y=point[1], z=0),
                    project_to_road=True,
                ).transform.location  # project to road to approximate Z
                path_transforms.append(
                    Transform(
                        location=Location(x=point[0], y=point[1], z=p_loc.z),
                        rotation=Rotation(),
                    ))
            waypoints = deque(path_transforms)
            waypoints.extend(
                itertools.islice(self._waypoints, self._wp_index,
                                 len(self._waypoints))
            )  # add the remaining global route for future
        else:
            waypoints = self._waypoints

        # construct waypoints message
        waypoints = collections.deque(
            itertools.islice(waypoints, 0,
                             DEFAULT_NUM_WAYPOINTS))  # only take 50 meters
        next_waypoint = waypoints[self._wp_index]
        wp_steer_speed_vector, wp_steer_speed_angle = \
            get_waypoint_vector_and_angle(
                next_waypoint, vehicle_transform
            )
        output_msg = WaypointsMessage(timestamp,
                                      waypoints=waypoints,
                                      wp_angle=wp_steer_speed_angle,
                                      wp_vector=wp_steer_speed_vector,
                                      wp_angle_speed=wp_steer_speed_angle)

        # send waypoints message
        waypoints_stream.send(output_msg)
        waypoints_stream.send(erdos.WatermarkMessage(timestamp))

    def _build_obstacle_map(self, vehicle_transform):
        """
        Construct an obstacle map given vehicle_transform.

        Args:
            vehicle_transform: pylot.utils.Transform of vehicle from can_bus
                stream

        Returns:
            an obstacle map that maps
                {id_time: (obstacle_origin, obstacle_range)}
            only obstacles within DEFAULT_DISTANCE_THRESHOLD in front of the
            ego vehicle are considered to save computation cost.
        """
        obstacle_map = {}
        prediction_msg = self._prediction_msgs.popleft()
        # look over all predictions
        for prediction in prediction_msg.predictions:
            time = 0
            # use all prediction times as potential obstacles
            for transform in prediction.trajectory:
                if vehicle_transform.location.is_within_distance_ahead(
                        transform.location, DEFAULT_DISTANCE_THRESHOLD):
                    # compute the obstacle origin and range of the obstacle
                    obstacle_origin = ((transform.location.x -
                                        DEFAULT_OBSTACLE_LENGTH / 2,
                                        transform.location.y -
                                        DEFAULT_OBSTACLE_WIDTH / 2),
                                       (DEFAULT_OBSTACLE_LENGTH,
                                        DEFAULT_OBSTACLE_WIDTH))
                    obs_id = str("{}_{}".format(prediction.id, time))
                    obstacle_map[obs_id] = obstacle_origin
                time += 1
        return obstacle_map

    def _compute_target_location(self, vehicle_transform):
        """
        Update the global waypoint route and compute the target location for
        RRT* search to plan for.

        Args:
            vehicle_transform: pylot.utils.Transform of vehicle from can_bus
                stream

        Returns:
            target location
        """
        ego_location = vehicle_transform.location.as_carla_location()
        self._waypoints = self._hd_map.compute_waypoints(
            ego_location, self._goal_location)
        target_waypoint = self._waypoints[self._wp_index]
        target_location = target_waypoint.location
        return target_location

    @staticmethod
    def _run_rrt_star(vehicle_transform, target_location, obstacle_map):
        """
        Run the RRT* algorithm given the vehicle_transform, target_location,
        and obstacle_map.

        Args:
            vehicle_transform: pylot.utils.Transform of vehicle from can_bus
                stream
            target_location: Location target
            obstacle_map: an obstacle map that maps
                {id_time: (obstacle_origin, obstacle_range)}

        Returns:
            np.ndarray, float
            return the path in form [[x0, y0],...] and final cost
            if solution not found, returns the path to the closest point to the
            target space and final cost is none
        """
        starting_state = (vehicle_transform.location.x,
                          vehicle_transform.location.y)
        target_space = ((target_location.x - DEFAULT_TARGET_LENGTH / 2,
                         target_location.y - DEFAULT_TARGET_WIDTH / 2),
                        (DEFAULT_TARGET_LENGTH, DEFAULT_TARGET_WIDTH))
        state_space = start_target_to_space(starting_state, target_space,
                                            DEFAULT_TARGET_LENGTH,
                                            DEFAULT_TARGET_WIDTH)
        path, cost = apply_rrt_star(state_space=state_space,
                                    starting_state=starting_state,
                                    target_space=target_space,
                                    obstacle_map=obstacle_map)
        return path, cost
