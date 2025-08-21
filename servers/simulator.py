import grpc
from concurrent import futures
import time
import math
import logging
import signal
from datetime import datetime, timezone
from typing import Iterator, List, Optional

from threading import Thread, Event

import proto.robot_control_pb2 as rc
import proto.robot_control_pb2_grpc as rc_grpc
import google.protobuf.timestamp_pb2 as g_timestamp

from PIL import Image

# 配置日志系统
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('robot_control.log')
    ]
)
simulator_logger = logging.getLogger('RobotSimulator')
service_logger = logging.getLogger("RobotControlService")


def get_proto_timestamp() -> g_timestamp.Timestamp:
    pb_timestamp = g_timestamp.Timestamp()
    pb_timestamp.FromDatetime(datetime.now(timezone.utc))
    return pb_timestamp


class RobotSimulator:
    """模拟机器人物理行为和状态管理"""

    def __init__(self):
        self.position = rc.Position(x=0.0, y=0.0, z=0.0)
        self.orientation = rc.Orientation(yaw=0.0, pitch=0.0, roll=0.0)
        self.battery_level = 1.0
        self.warnings: List[str] = []
        self.state_code: rc.RobotState.StateEnum = rc.RobotState.IDLE
        self._target: Optional[rc.TargetPosition] = None
        self._max_speed = 1
        self._tolerance = 0.05
        self._last_update_time = time.monotonic()

        self.object_held = ""
        self.is_using_gripper = False
        self.gripper_work_time_cost = 10    # second(s)
        self.gripper_work_timer = 0         # > 0 <=> is_moving_arm

        self.update_interval = 0.25  # second(s)
        self.update_worker_running_flag = True
        self.update_worker_stop_grace_period = 5
        self.update_worker = Thread(target=self.__state_update_worker)
        self.update_worker.start()
        simulator_logger.info("Simulator update worker thread started")

        simulator_logger.info("RobotSimulator initialized")
        simulator_logger.info(
            f"Initial position: ({self.position.x:.2f}, {self.position.y:.2f}, {self.position.z:.2f})")
        simulator_logger.info(f"Battery level: {self.battery_level:.2%}")

    @property
    def is_moving_arm(self) -> bool:
        return self.gripper_work_timer > 0

    def __state_update_worker(self):
        while self.update_worker_running_flag:
            current_time = time.monotonic()
            self.__update_states(current_time)
            time.sleep(self.update_interval)
        simulator_logger.info("State update worker thread exited")

    def __update_states(self, current_time: float):
        """根据目标位置更新机器人各种状态（物理模拟）"""

        # 计算时间增量（自上次更新以来的时间）
        dt = current_time - self._last_update_time
        self._last_update_time = current_time

        # 先更新夹爪
        if self.is_moving_arm:
            self.gripper_work_timer -= dt
            if not self.is_moving_arm:
                if len(self.object_held) == 0:
                    self.is_using_gripper = False
                    simulator_logger.info("Gripper place (async) finished!")
                else:
                    self.is_using_gripper = True
                    simulator_logger.info("Gripper pick-up (async) finished!")

        if (not self._target or self.state_code != rc.RobotState.MOVING
                or self.state_code == rc.RobotState.EMERGE_STOP):
            return

        # 计算当前位置到目标的向量
        dx = self._target.position.x - self.position.x
        dy = self._target.position.y - self.position.y
        dz = self._target.position.z - self.position.z

        # 计算距离和方向
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        if distance < self._tolerance:
            simulator_logger.info("Target reached within tolerance")
            self.state_code = rc.RobotState.IDLE
            return

        # 计算移动步长
        step = min(distance, self._max_speed * dt)
        ratio = step / distance if distance > 0 else 0

        # 记录移动前的状态
        prev_pos = (self.position.x, self.position.y, self.position.z)

        # 更新位置
        self.position.x += dx * ratio
        self.position.y += dy * ratio
        self.position.z += dz * ratio

        # 记录移动
        simulator_logger.debug(
            f"Moved {step:.4f}m from ({prev_pos[0]:.2f}, {prev_pos[1]:.2f}, {prev_pos[2]:.2f}) "
            f"to ({self.position.x:.2f}, {self.position.y:.2f}, {self.position.z:.2f}) "
            f"(dt={dt:.4f}s, speed={self._max_speed:.2f}m/s)"
        )

        # 更新电池（每次移动消耗）
        prev_battery = self.battery_level
        self.battery_level = max(0.0, self.battery_level - 0.0001 * step)
        battery_change = prev_battery - self.battery_level

        if battery_change > 0:
            simulator_logger.debug(
                f"Battery decreased: {prev_battery:.2%} → {self.battery_level:.2%} "
                f"(-{battery_change:.6%})"
            )

        if self.battery_level < 0.2 and "Low battery" not in self.warnings:
            self.warnings.append("Low battery")
            simulator_logger.warning("Low battery warning activated")

    def set_target(self, target: rc.TargetPosition):
        """设置新的目标位置"""
        if self.state_code == rc.RobotState.MOVING:
            simulator_logger.warning("Cannot move a moving robot. Ignored")
            return
        self._target = target
        self._max_speed = target.max_speed
        self._tolerance = target.tolerance
        self.state_code = rc.RobotState.MOVING
        self._last_update_time = time.monotonic()

        simulator_logger.info(
            f"New target set: ({target.position.x:.2f}, {target.position.y:.2f}, {target.position.z:.2f}) "
            f"with max speed {target.max_speed:.2f}m/s and tolerance {target.tolerance:.3f}m"
        )
        simulator_logger.info(f"Movement started")

    def move_relative(self, direction: rc.RobotDirection) -> rc.TargetPosition:
        """小位移微调位置"""
        if self.state_code == rc.RobotState.MOVING:
            simulator_logger.warning("Trying to move_tweak while moving. Ignored")
            # 此时 target 不可能是 None
            return self._target
        target_pos = self.position
        sqrt2_inv = 1.414 / 2
        match direction.direction:
            case rc.RobotDirection.FORWARD_LEFT:
                target_pos.x += direction.distance * sqrt2_inv
                target_pos.y += direction.distance * sqrt2_inv
            case rc.RobotDirection.FORWARD_RIGHT:
                target_pos.x += direction.distance * sqrt2_inv
                target_pos.y -= direction.distance * sqrt2_inv
            case rc.RobotDirection.BACKWARD_LEFT:
                target_pos.x -= direction.distance * sqrt2_inv
                target_pos.y += direction.distance * sqrt2_inv
            case rc.RobotDirection.BACKWARD_RIGHT:
                target_pos.x -= direction.distance * sqrt2_inv
                target_pos.y -= direction.distance * sqrt2_inv
            case rc.RobotDirection.FORWARD:
                target_pos.x += direction.distance
            case rc.RobotDirection.BACKWARD:
                target_pos.x -= direction.distance
        target = rc.TargetPosition(
            position=target_pos,
            orientation=self.orientation,
            max_speed=self._max_speed,
            tolerance=self._tolerance,
        )
        self.set_target(target)
        return target

    def emergency_stop(self):
        """紧急停止机器人"""
        self.state_code = rc.RobotState.EMERGE_STOP
        self.is_using_gripper = False

        if "Emergency stop activated" not in self.warnings:
            self.warnings.append("Emergency stop activated")
            simulator_logger.critical("EMERGENCY STOP ACTIVATED!")

    def pickup_object(self, hint: str) -> bool:
        """夹起镜头前的指定物品。如果夹爪不可用或不空闲，则失败并返回 False"""
        if len(self.object_held) == 0:
            self.object_held = hint
            simulator_logger.info(f"Gripper is trying to pick up object '{hint}'...")
            self.gripper_work_timer = self.gripper_work_time_cost
            return True
        else:
            simulator_logger.warning(
                f"Gripper is not available! (object_held='{self.object_held}', "
                f"is_gripper_moving={self.is_using_gripper})")
            return False

    def place_object(self, hint: str) -> bool:
        """放下夹爪上的物品"""
        if len(self.object_held) == 0:
            simulator_logger.warning("Gripper is empty! Do nothing.")
            return False
        else:
            self.object_held = ""
            simulator_logger.info(
                f"Gripper is trying to place object '{self.object_held}' with hint '{hint}'.")
            self.gripper_work_timer = self.gripper_work_time_cost
            return True

    def get_state(self) -> rc.RobotState:
        """获取当前机器人状态"""
        # 创建正确的时间戳对象
        pb_timestamp = get_proto_timestamp()

        return rc.RobotState(
            position=self.position,
            orientation=self.orientation,
            battery_level=self.battery_level,
            warnings=self.warnings,
            state_code=self.state_code,
            is_using_gripper=self.is_using_gripper,
            is_moving_arm=self.is_moving_arm,
            timestamp=pb_timestamp
        )

    def get_camera_image(self) -> rc.ImageResponse:
        with open("../mock/camera.jpg", mode="rb") as img:
            img_bytes = img.read()
            if not img_bytes:
                img_bytes = b''
            pil_img = Image.open("../mock/camera.jpg")
            meta = rc.ImageMetadata(
                capture_time=get_proto_timestamp(),
                width=pil_img.width,
                height=pil_img.height,
                channels=3,
                camera_position=self.position,
                camera_orientation=self.orientation
            )
            pil_img.close()
            camera_conf = rc.CameraConfig(
                format=rc.CameraConfig.JPEG,
                resolution=rc.CameraConfig.RES_UNKNOWN,
                frame_rate=30,
                auto_exposure=True,
                exposure_time=0.25,
                gain=1,
                color=True,
                quality=1,
                enable_depth=False,
                enable_pointcloud=False
            )
            return rc.ImageResponse(
                metadata=meta,
                config=camera_conf,
                full_image=img_bytes
            )

    def stop(self):
        """停止模拟器并回收资源"""
        simulator_logger.info("Stopping RobotSimulator...")
        self.update_worker_running_flag = False
        if self.update_worker.is_alive():
            self.update_worker.join(timeout=self.update_worker_stop_grace_period)
            if self.update_worker.is_alive():
                simulator_logger.warning("State update worker thread did not exit gracefully")
            else:
                simulator_logger.info("State update worker thread joined successfully")
        simulator_logger.info("RobotSimulator stopped")


class RobotControlServicer(rc_grpc.RobotControlServiceServicer):
    def __init__(self, robot: RobotSimulator):
        self.robot = robot

        self.stream_update_interval = 0.25

    def MoveToPosition(self, request: rc.TargetPosition, context) -> rc.ControlResponse:
        service_logger.info("Received move_to_position command")
        service_logger.debug(f"Request: {request}")
        try:
            # 设置目标位置
            self.robot.set_target(request)

            # 等待移动完成 (简化实现，实际中应有超时和状态检查)
            service_logger.info("Waiting for robot movement finishing...")
            while True:
                current_state = self.robot.get_state()
                if current_state.state_code != rc.RobotState.MOVING:
                    break
                service_logger.debug(
                    f"Checking robot position: x={current_state.position.x}, "
                    f"y={current_state.position.y}, is_moving={current_state.state_code == rc.RobotState.MOVING}")
                time.sleep(0.5)

            # 检查是否成功到达
            service_logger.info("Checking robot status after waiting...")
            current_state = self.robot.get_state()
            success = self._check_position_reached(
                current_state.position,
                request.position,
                request.tolerance
            ) and self._check_orientation_reached(
                current_state.orientation,
                request.orientation,
                request.orientation_tolerance
            )

            return self._build_response(
                rc.ControlResponse.SUCCESS if success else rc.ControlResponse.PARTIAL_SUCCESS,
                "Movement completed" if success else "Partial movement achieved",
                current_state
            )

        except Exception as e:
            return self._build_response(
                rc.ControlResponse.INVALID_TARGET,
                f"Movement failed: {str(e)}",
                self.robot.get_state()
            )

    def StreamMoveToPosition(self, request: rc.TargetPosition, context) -> Iterator[rc.RobotState]:
        service_logger.info("Received move_to_position command (stream)")
        try:
            self.robot.set_target(request)

            state: rc.RobotState
            # 持续发送状态直到到达目标
            while True:
                state = self.robot.get_state()
                # 检查是否到达目标
                if state.state_code != rc.RobotState.MOVING or self._position_reached(state, request):
                    break

                yield state
                time.sleep(self.stream_update_interval)

            # yield state state_code != MOVING indicating the end of the stream
            # 防止一种情况：position reached 但 MOVING，下一轮更新就会变为 IDLE
            if self._position_reached(state, request):
                state.state_code = rc.RobotState.IDLE
            yield state

        except Exception as e:
            # 发送错误状态
            error_state = self.robot.get_state()
            error_state.warnings.append(f"Streaming error at MoveToPosition: {str(e)}")
            yield error_state

    def StreamMove(self, request: rc.RobotDirection, context) -> Iterator[rc.RobotState]:
        service_logger.info("Received move_relative command (stream)")
        try:
            target_pos = self.robot.move_relative(request)

            state: rc.RobotState
            while True:
                state = self.robot.get_state()
                if state.state_code != rc.RobotState.MOVING or self._position_reached(state, target_pos):
                    break

                yield state
                time.sleep(self.stream_update_interval)

            # yield state state_code != MOVING indicating the end of the stream
            # 防止一种情况：position reached 但 MOVING，下一轮更新就会变为 IDLE
            if self._position_reached(state, target_pos):
                state.state_code = rc.RobotState.IDLE
            yield state

        except Exception as e:
            error_state = self.robot.get_state()
            error_state.warnings.append(f"Streaming error at MoveRelative: {str(e)}")
            yield error_state

    def GetCurrentPosition(self, request: rc.Empty, context) -> rc.ControlResponse:
        service_logger.info("Received get_current_position command")
        try:
            state = self.robot.get_state()
            return self._build_response(
                rc.ControlResponse.SUCCESS,
                "Current state retrieved",
                state
            )
        except Exception as e:
            return self._build_response(
                rc.ControlResponse.EMERGENCY_STOP,
                f"Failed to get state: {str(e)}",
                rc.RobotState()  # 返回空状态
            )

    def EmergencyStop(self, request: rc.Empty, context) -> rc.ControlResponse:
        service_logger.info("Received emergency_stop command")
        try:
            self.robot.emergency_stop()
            return self._build_response(
                rc.ControlResponse.SUCCESS,
                "Emergency stop activated",
                self.robot.get_state()
            )
        except Exception as e:
            return self._build_response(
                rc.ControlResponse.EMERGENCY_STOP,
                f"Emergency stop failed: {str(e)}",
                self.robot.get_state()
            )

    def StreamPickUpObject(self, request: rc.PickOrPlaceCmd, context) -> Iterator[rc.RobotState]:
        if self.robot.get_state().state_code == rc.RobotState.MOVING:
            service_logger.warning("Trying to pick up object while moving!")
        service_logger.info(f"Received pick_up_object command (stream, cmd={request.cmd})")
        try:
            if not self.robot.pickup_object(request.cmd):
                raise RuntimeError("Gripper is already in use or not available")

            state: rc.RobotState
            while True:
                state = self.robot.get_state()
                if not state.is_moving_arm:
                    break

                yield state
                time.sleep(self.stream_update_interval)

            # yield state is_moving_arm==False indicating the end of the stream
            yield state
        except Exception as e:
            error_state = self.robot.get_state()
            error_state.warnings.append(f"Streaming error at PickUpObject: {str(e)}")
            yield error_state

    def StreamPlaceObject(self, request: rc.PickOrPlaceCmd, context) -> Iterator[rc.RobotState]:
        if self.robot.get_state().state_code == rc.RobotState.MOVING:
            service_logger.warning("Trying to place object while moving!")
        service_logger.info(f"Received place_object command (stream, cmd={request.cmd})")
        try:
            if not self.robot.place_object(request.cmd):
                raise RuntimeError("Gripper is empty or not available")

            state: rc.RobotState
            while True:
                state = self.robot.get_state()
                if not state.is_moving_arm:
                    break

                yield state
                time.sleep(self.stream_update_interval)

            # yield state is_moving_arm==False indicating the end of the stream
            yield state
        except Exception as e:
            error_state = self.robot.get_state()
            error_state.warnings.append(f"Streaming error at PlaceObject: {str(e)}")
            yield error_state

    def _position_reached(self, state: rc.RobotState, target: rc.TargetPosition) -> bool:
        return (
            self._check_position_reached(
                state.position,
                target.position,
                target.tolerance)
            and self._check_orientation_reached(
                state.orientation,
                target.orientation,
                target.orientation_tolerance)
        )

    @staticmethod
    def _check_position_reached(current: rc.Position, target: rc.Position, tolerance: float) -> bool:
        dx = current.x - target.x
        dy = current.y - target.y
        dz = current.z - target.z
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return distance <= tolerance

    @staticmethod
    def _check_orientation_reached(current: rc.Orientation, target: rc.Orientation, tolerance: float) -> bool:
        return (
                abs(current.yaw - target.yaw) <= tolerance and
                abs(current.pitch - target.pitch) <= tolerance and
                abs(current.roll - target.roll) <= tolerance
        )

    def _build_response(self, code: rc.ControlResponse.ResultCode, message: str,
                        state: rc.RobotState) -> rc.ControlResponse:
        return rc.ControlResponse(
            code=code,
            message=message,
            current_state=state,
            suggested_actions=self._get_suggested_actions(code)
        )

    @staticmethod
    def _get_suggested_actions(code: rc.ControlResponse.ResultCode) -> list[str]:
        suggestions = {
            rc.ControlResponse.LOW_BATTERY: ["Return to charging station"],
            rc.ControlResponse.OBSTACLE_DETECTED: ["Scan environment", "Find alternative path"],
            rc.ControlResponse.OUT_OF_RANGE: ["Check target coordinates", "Verify workspace limits"],
            rc.ControlResponse.INVALID_TARGET: ["Validate target parameters", "Check obstacle map"]
        }
        return suggestions.get(code, [])


class CameraServicer(rc_grpc.CameraServiceServicer):
    def __init__(self, robot: RobotSimulator):
        self.robot = robot

    def GetImage(self, request: rc.Empty, context) -> rc.ImageResponse:
        return self.robot.get_camera_image()

    def GetCameraConfig(self, request: rc.Empty, context) -> rc.CameraConfig:
        # 假设模拟器可以返回配置
        image_resp = self.robot.get_camera_image()
        return image_resp.config


def serve():
    robot_sim = RobotSimulator()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    rc_grpc.add_RobotControlServiceServicer_to_server(
        RobotControlServicer(robot_sim), server)
    rc_grpc.add_CameraServiceServicer_to_server(
        CameraServicer(robot_sim), server)

    server.add_insecure_port('[::]:50051')
    server.start()
    service_logger.info("gRPC server started on port 50051")

    # 信号处理：捕获终止信号并触发清理
    stop_event = Event()

    def handle_termination(signum, _frame):
        signal_name = signal.Signals(signum).name
        service_logger.info(f"Received {signal_name} signal, initiating shutdown...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_termination)
    signal.signal(signal.SIGTERM, handle_termination)

    # 阻塞等待终止信号
    stop_event.wait()

    service_logger.info("Starting resource cleanup...")
    robot_sim.stop()
    server.stop(5)
    service_logger.info("All resources cleaned up. Bye :)")


if __name__ == '__main__':
    serve()
