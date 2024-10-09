import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class LuxsanMofo:
    def __init__(self):
        pass

    def connect(self):
        pass

    def run_calibration(self):
        pass

    def teleop_step(self, record_data=False):
        pass

    def capture_observation(self):
        pass

    def send_action(self, action):
        pass

    def disconnect(self):
        pass


class LuxsanMofoSimulator:
    def __init__(self, episode, root, repo_id):
        self.dataset = LeRobotDataset(repo_id, root=root)
        self.from_idx = self.dataset.episode_data_index["from"][episode].item()
        self.to_idx = self.dataset.episode_data_index["to"][episode].item()
        self.counter = 0
        self.action_history = []
        self.fig, self.ax = plt.subplots()
        self.lines = [self.ax.plot([], [], "r-")[0] for _ in range(6)]
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-2, 2)
        plt.ion()
        plt.show()

    def connect(self):
        pass

    def run_calibration(self):
        pass

    def teleop_step(self, record_data=False):
        pass

    def capture_observation(self):
        if self.counter + self.from_idx < self.to_idx:
            observation = {
                "observation.images.camera_1": self.dataset[self.counter + self.from_idx][
                    "observation.images.camera_1"
                ],
                "observation.images.camera_2": self.dataset[self.counter + self.from_idx][
                    "observation.images.camera_2"
                ],
                "observation.state": self.dataset[self.counter + self.from_idx]["observation.state"],
            }
            self.counter += 1
            return observation
        else:
            raise ValueError("Episode is over")

    def send_action(self, action):
        print(action)
        self.action_history.append(action[0])
        if len(self.action_history) > 100:
            self.action_history.pop(0)
        for i in range(6):
            self.lines[i].set_xdata(np.arange(len(self.action_history)))
            self.lines[i].set_ydata([x[i] for x in self.action_history])
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

        # Add labels and different colors
        labels = ["Joint 0", "Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5"]
        colors = ["r", "g", "b", "c", "m", "y"]
        for i in range(6):
            self.lines[i].set_label(labels[i])
            self.lines[i].set_color(colors[i])
        plt.legend()

    def disconnect(self):
        plt.savefig("joint_movement.png")


def run(mofo: LuxsanMofo, policy: Policy, args):
    start_time = time.perf_counter()
    while True:
        try:
            print("Start run luxan mofo, press CTRL+C to stop")
            start_loop_t = time.perf_counter()
            with torch.inference_mode():
                try:
                    observation = mofo.capture_observation()
                except ValueError as e:
                    print(e)
                    break

                observation = {
                    key: observation[key].unsqueeze(0).to(args.device, non_blocking=True) for key in observation
                }

                action = policy.select_action(observation)
                action = action.to("cpu").numpy()  # Convert to CPU / numpy.
                assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"
                mofo.send_action(action)
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / args.fps - dt_s)
        except KeyboardInterrupt:
            print("Stop the luxsan mofo")
            break

        print(f"Time elapsed: {time.perf_counter() - start_time:.2f} seconds")


def main():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="simulator")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--root", type=str)  # --root is only used in simulator mode
    parser.add_argument("--repo_id", type=str)  # --repo_id is only used in simulator mode
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    print(
        f"""
    Arguments:
        mode: {args.mode}
        model_checkpoint: {args.model_checkpoint}
        root: {args.root}
        repo_id: {args.repo_id}
        device: {args.device}
        fps: {args.fps}
    """
    )
    policy = ACTPolicy.from_pretrained(args.model_checkpoint)
    if args.mode == "simulator":
        mofo = LuxsanMofoSimulator(episode=1, root=args.root, repo_id=args.repo_id)
    elif args.mode == "real":
        mofo = LuxsanMofo()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    mofo.connect()
    run(mofo=mofo, policy=policy, args=args)
    mofo.disconnect()


if __name__ == "__main__":
    main()
