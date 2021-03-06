import erdos
import json
import os


class TrajectoryLoggerOperator(erdos.Operator):
    """ Logs tracked obstacle trajectories."""
    def __init__(self,
                 obstacle_tracking_stream,
                 name,
                 flags,
                 log_file_name=None):
        obstacle_tracking_stream.add_callback(self.on_trajectories_msg)
        self._name = name
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._flags = flags
        self._msg_cnt = 0

    @staticmethod
    def connect(obstacle_tracking_stream):
        return []

    def on_trajectories_msg(self, msg):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        self._msg_cnt += 1
        if self._msg_cnt % self._flags.log_every_nth_message != 0:
            return
        trajectories = [
            str(trajectory) for trajectory in msg.obstacle_trajectories
        ]
        assert len(msg.timestamp.coordinates) == 1
        timestamp = msg.timestamp.coordinates[0]
        # Write the trajectories.
        file_name = os.path.join(self._flags.data_path,
                                 'trajectories-{}.json'.format(timestamp))
        with open(file_name, 'w') as outfile:
            json.dump(trajectories, outfile)
