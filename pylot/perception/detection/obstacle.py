import numpy as np

import pylot.utils
from pylot.perception.detection.utils import BoundingBox3D, \
    get_bounding_box_in_camera_view


class Obstacle(object):
    """ An Obstacle represents a dynamic obstacle that we could encounter on the
    road. This class provides helper functions to detect obstacles and provide
    bounding boxes for them.
    """
    def __init__(self, id, label, transform, bounding_box, forward_speed):
        """ Initialize an obstacle.

        Args:
            id: The id of the obstacle.
            label: The label of the obstacle.
            transform: The transform of the obstacle.
            bounding_box: A perception.detection.utils.BoundingBox3D of the
                  obstacle.
            forward_speed: The forward speed of the obstacle.
        """
        self.id = id
        self.transform = transform
        self.bounding_box = bounding_box
        self.forward_speed = forward_speed
        self.label = label
        if label == 'vehicle':
            self.segmentation_class = 10
        elif label == 'person':
            self.segmentation_class = 4
        else:
            raise ValueError('label should be: vehicle or person')
        # Thresholds to be used for detection of the obstacle.
        self.__segmentation_threshold = 0.20
        self.__depth_threshold = 5

    @classmethod
    def from_carla_actor(cls, actor):
        """ Creates an Obstacle from a carla actor.

        Args:
            actor: The actor to initialize the obstacle with. (should be of
            type carla.Vehicle or carla.Walker)
        """
        import carla
        if not isinstance(actor, (carla.Vehicle, carla.Walker)):
            raise ValueError("The actor should be of type carla.Vehicle or "
                             "carla.Walker to initialize the Obstacle class.")
        # Convert the transform provided by the simulation to the Pylot class.
        transform = pylot.utils.Transform.from_carla_transform(
            actor.get_transform())
        # Convert the bounding box from the simulation to the Pylot one.
        bounding_box = BoundingBox3D.from_carla_bounding_box(
            actor.bounding_box)
        # Get the speed of the obstacle.
        velocity_vector = pylot.utils.Vector3D.from_carla_vector(
            actor.get_velocity())
        forward_speed = velocity_vector.magnitude()
        if isinstance(actor, carla.Vehicle):
            label = 'vehicle'
        else:
            label = 'person'
        # TODO (Sukrit): Move from vehicles and people to separate classes
        # for bicycles, motorcycles, cars and persons.
        return cls(actor.id, label, transform, bounding_box, forward_speed)

    def distance(self, vehicle_transform):
        """ Returns the distance of the obstacle from the vehicle represented
        by the vehicle transform.

        The distance provides an estimate of the depth returned by the depth
        camera sensor in Carla. As a result, the distance is defined as the
        displacement of the obstacle along either the X or the Y axis.

        Args:
            vehicle_transform: The pylot.utils.Transform of the vehicle to find
            the distance of the obstacle from.

        Returns:
            The distance (in metres) of the obstacle from the vehicle.
        """
        import numpy as np

        # Get the location of the vehicle and the obstacle as numpy arrays.
        vehicle_location = vehicle_transform.location.as_numpy_array()
        obstacle_location = self.transform.location.as_numpy_array()

        # Calculate the vector from the vehicle to the obstacle.
        # Scale it by the forward vector, and calculate the norm.
        relative_vector = vehicle_location - obstacle_location
        distance = np.linalg.norm(
            relative_vector *
            vehicle_transform.forward_vector.as_numpy_array())
        return distance

    def to_camera_view(self, depth_frame, segmented_frame):
        """ Retrieves the 2D bounding box for the obstacle with respect to the
        given camera setup.

        Heuristically uses the depth frame and segmentation frame to figure out
        if the obstacle is in view of the camera or not.

        Args:
            depth_frame: The DepthFrame to be used to compare the depth
                to the distance of the obstacle from the sensor.
            segmented_frame: The SegmentedFrame to be used to compare
                the segmentation class.

        Returns:
            A BoundingBox2D instance representing a rectangle over the obstacle
            if the obstacle is deemed to be visible, None otherwise.
        """
        # Convert the bounding box of the obstacle to the camera coordinates.
        bb_coordinates = self.bounding_box.to_camera_view(
            self.transform, depth_frame.camera_setup.get_extrinsic_matrix(),
            depth_frame.camera_setup.get_intrinsic_matrix())

        # Threshold the bounding box to be within the camera view.
        bbox_2d = get_bounding_box_in_camera_view(
            bb_coordinates, depth_frame.camera_setup.width,
            depth_frame.camera_setup.height)
        if not bbox_2d:
            return None
        # Crop the segmented and depth image to the given bounding box.
        cropped_image = segmented_frame.as_numpy_array()[
            bbox_2d.y_min:bbox_2d.y_max, bbox_2d.x_min:bbox_2d.x_max]
        cropped_depth = depth_frame.as_numpy_array()[
            bbox_2d.y_min:bbox_2d.y_max, bbox_2d.x_min:bbox_2d.x_max]

        # If the size of the bounding box is greater than 0, ensure that the
        # bounding box contains more than a threshold of pixels corresponding
        # to the required segmentation class.
        if cropped_image.size > 0:
            masked_image = np.zeros_like(cropped_image)
            masked_image[np.where(
                cropped_image == self.segmentation_class)] = 1
            seg_threshold = self.__segmentation_threshold * masked_image.size
            if np.sum(masked_image) >= seg_threshold:
                # The bounding box contains the required number of pixels that
                # belong to the required class. Ensure that the depth of the
                # obstacle is the depth in the image.
                masked_depth = cropped_depth[np.where(masked_image == 1)]
                mean_depth = np.mean(masked_depth) * 1000
                depth = self.distance(depth_frame.camera_setup.get_transform())
                if abs(depth - mean_depth) <= self.__depth_threshold:
                    return bbox_2d
        return None
