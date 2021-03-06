import erdos


class CameraVisualizerOperator(erdos.Operator):
    """ Subscribes to camera streams, and visualizes frames."""
    def __init__(self, camera_stream, name, log_file_name=None):
        camera_stream.add_callback(self.display_frame)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)

    @staticmethod
    def connect(camera_stream):
        return []

    def display_frame(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        msg.frame.visualize(self._name, msg.timestamp)
