import math
import numpy as np

from pylot.perception.detection.traffic_light import TrafficLightColor


def get_world_vec_dist(x_dst, y_dst, x_src, y_src):
    vec = np.array([x_dst, y_dst] - np.array([x_src, y_src]))
    dist = math.sqrt(vec[0]**2 + vec[1]**2)
    if abs(dist) < 0.00001:
        return vec, dist
    else:
        return vec / dist, dist


def get_angle(vec_dst, vec_src):
    angle = (math.atan2(vec_dst[1], vec_dst[0]) -
             math.atan2(vec_src[1], vec_src[0]))
    if angle > math.pi:
        angle -= 2 * math.pi
    elif angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _is_person_on_hit_zone(p_dist, p_angle, flags):
    return (math.fabs(p_angle) < flags.person_angle_hit_thres
            and p_dist < flags.person_distance_hit_thres)


def _is_person_on_near_hit_zone(p_dist, p_angle, flags):
    return (math.fabs(p_angle) < flags.person_angle_emergency_thres
            and p_dist < flags.person_distance_emergency_thres)


def stop_person(ego_vehicle_location, person_location, wp_vector,
                speed_factor_p, flags):
    """ Computes a stopping factor for ego vehicle given a person pos.

    Args:
        ego_vehicle_location: Location of the ego vehicle in world coordinates.
        person_location: Location of the person in world coordinates.
        flags: A absl flags object.

    Returns:
        A stopping factor between 0 and 1 (i.e., no braking).
    """
    speed_factor_p_temp = 1
    p_vector, p_dist = get_world_vec_dist(person_location.x, person_location.y,
                                          ego_vehicle_location.x,
                                          ego_vehicle_location.y)
    p_angle = get_angle(p_vector, wp_vector)
    if _is_person_on_hit_zone(p_dist, p_angle, flags):
        speed_factor_p_temp = p_dist / (flags.coast_factor *
                                        flags.person_distance_hit_thres)
    if _is_person_on_near_hit_zone(p_dist, p_angle, flags):
        speed_factor_p_temp = 0
    if (speed_factor_p_temp < speed_factor_p):
        speed_factor_p = speed_factor_p_temp
    return speed_factor_p


def stop_vehicle(ego_vehicle_location, obs_vehicle_location, wp_vector,
                 speed_factor_v, flags):
    """ Computes a stopping factor for ego vehicle given a vehicle pos.

    Args:
        ego_vehicle_location: Location of the ego vehicle in world coordinates.
        obs_vehicle_location: Location of the vehicle in world coordinates.
        flags: A absl flags object.

    Returns:
        A stopping factor between 0 and 1 (i.e., no braking).
    """
    speed_factor_v_temp = 1
    v_vector, v_dist = get_world_vec_dist(obs_vehicle_location.x,
                                          obs_vehicle_location.y,
                                          ego_vehicle_location.x,
                                          ego_vehicle_location.y)
    v_angle = get_angle(v_vector, wp_vector)

    min_angle = -0.5 * flags.vehicle_angle_thres / flags.coast_factor
    max_angle = flags.vehicle_angle_thres / flags.coast_factor
    max_dist = flags.vehicle_distance_thres * flags.coast_factor
    medium_max_angle = flags.vehicle_angle_thres
    medium_dist = flags.vehicle_distance_thres
    if ((min_angle < v_angle < max_angle and v_dist < max_dist) or
        (min_angle < v_angle < medium_max_angle and v_dist < medium_dist)):
        speed_factor_v_temp = v_dist / (flags.coast_factor *
                                        flags.vehicle_distance_thres)

    min_nearby_angle = -0.5 * flags.vehicle_angle_thres * flags.coast_factor
    max_nearby_angle = flags.vehicle_angle_thres * flags.coast_factor
    nearby_dist = flags.vehicle_distance_thres / flags.coast_factor
    if (min_nearby_angle < v_angle < max_nearby_angle
            and v_dist < nearby_dist):
        speed_factor_v_temp = 0

    if speed_factor_v_temp < speed_factor_v:
        speed_factor_v = speed_factor_v_temp

    return speed_factor_v


def stop_traffic_light(ego_vehicle_location, tl_location, tl_state, wp_vector,
                       wp_angle, speed_factor_tl, flags):
    """ Computes a stopping factor for ego vehicle given a traffic light.

    Args:
        ego_vehicle_location: Location of the ego vehicle in world coordinates.
        tl_location: Location of the traffic light in world coordinates.
        flags: A absl flags object.

    Returns:
        A stopping factor between 0 and 1 (i.e., no braking).
    """
    speed_factor_tl_temp = 1
    if (tl_state == TrafficLightColor.YELLOW
            or tl_state == TrafficLightColor.RED):
        tl_vector, tl_dist = get_world_vec_dist(tl_location.x, tl_location.y,
                                                ego_vehicle_location.x,
                                                ego_vehicle_location.y)
        tl_angle = get_angle(tl_vector, wp_vector)

        if ((0 < tl_angle <
             flags.traffic_light_angle_thres / flags.coast_factor and
             tl_dist < flags.traffic_light_max_dist_thres * flags.coast_factor)
                or (0 < tl_angle < flags.traffic_light_angle_thres
                    and tl_dist < flags.traffic_light_max_dist_thres)
                and math.fabs(wp_angle) < 0.2):

            speed_factor_tl_temp = tl_dist / (
                flags.coast_factor * flags.traffic_light_max_dist_thres)

        if ((0 < tl_angle <
             flags.traffic_light_angle_thres * flags.coast_factor and
             tl_dist < flags.traffic_light_max_dist_thres / flags.coast_factor)
                and math.fabs(wp_angle) < 0.2):
            speed_factor_tl_temp = 0

        if (speed_factor_tl_temp < speed_factor_tl):
            speed_factor_tl = speed_factor_tl_temp

    return speed_factor_tl


def stop_for_agents(ego_vehicle_location, wp_angle, wp_vector, wp_angle_speed,
                    obstacles, traffic_lights, hd_map, flags):
    speed_factor = 1
    speed_factor_tl = 1
    speed_factor_p = 1
    speed_factor_v = 1

    for obstacle in obstacles:
        if obstacle.label == 'vehicle' and flags.stop_for_vehicles:
            # Only brake for vehicles that are in ego vehicle's lane.
            if hd_map.are_on_same_lane(ego_vehicle_location,
                                       obstacle.transform.location):
                new_speed_factor_v = stop_vehicle(ego_vehicle_location,
                                                  obstacle.transform.location,
                                                  wp_vector, speed_factor_v,
                                                  flags)
                speed_factor_v = min(speed_factor_v, new_speed_factor_v)
        if obstacle.label == 'person' and flags.stop_for_people:
            # Only brake for people that are on the road.
            if hd_map.is_on_lane(obstacle.transform.location):
                new_speed_factor_p = stop_person(ego_vehicle_location,
                                                 obstacle.transform.location,
                                                 wp_vector, speed_factor_p,
                                                 flags)
                speed_factor_p = min(speed_factor_p, new_speed_factor_p)

    if flags.stop_for_traffic_lights:
        for tl in traffic_lights:
            if (hd_map.must_obbey_traffic_light(ego_vehicle_location,
                                                tl.transform.location)
                    and _is_traffic_light_visible(
                        ego_vehicle_location, tl.transform.location, flags)):
                new_speed_factor_tl = stop_traffic_light(
                    ego_vehicle_location, tl.transform.location, tl.state,
                    wp_vector, wp_angle, speed_factor_tl, flags)
                speed_factor_tl = min(speed_factor_tl, new_speed_factor_tl)

    speed_factor = min(speed_factor_tl, speed_factor_p, speed_factor_v)

    # Slow down around corners.
    if math.fabs(wp_angle_speed) < 0.1:
        speed_factor = 0.3 * speed_factor

    state = {
        'stop_person': speed_factor_p,
        'stop_vehicle': speed_factor_v,
        'stop_traffic_lights': speed_factor_tl
    }

    return speed_factor, state


def _is_traffic_light_visible(ego_vehicle_location, tl_location, flags):
    _, tl_dist = get_world_vec_dist(ego_vehicle_location.x,
                                    ego_vehicle_location.y, tl_location.x,
                                    tl_location.y)
    return tl_dist > flags.traffic_light_min_dist_thres
