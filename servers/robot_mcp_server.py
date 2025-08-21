#!/usr/bin/env python3
"""
MCP bridge for robot_control.v1 using fastmcp.

Run:  python robot_mcp_server.py --http-port 8000 --grpc-port 50051
"""

from __future__ import annotations

import asyncio.exceptions
import functools
import io
import os
import sys
import time
from typing import Callable, Dict, Any, Tuple

import grpc
from google.protobuf.json_format import MessageToDict

from fastmcp import FastMCP

import proto.robot_control_pb2 as pb
import proto.robot_control_pb2_grpc as pb_grpc

# import numpy as np
import base64
from PIL import Image

# It won't be written into requirements.txt
# because it should be ported to OpenHarmony platform
from ultralytics import YOLO

from models.paths import MODEL_PATH

import logging

# ────────────────────────────────────────────────────────────────────────────
# Logger configuration
# ────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("../robot_mcp_server.log")
    ]
)
logger = logging.getLogger("RobotMCP")

# ────────────────────────────────────────────────────────────────────────────
#  Environment / CLI configuration
# ────────────────────────────────────────────────────────────────────────────

GRPC_HOST_DEFAULT = "localhost"
GRPC_PORT_DEFAULT = 50051
GRPC_CHANNEL: grpc.aio.Channel | None = None

MIN_DIST_TOLERANCE = 0.01
MIN_ORIENT_TOLERANCE = 0.05
MIN_SPEED = 0.5


def get_channel() -> grpc.aio.Channel:
    global GRPC_CHANNEL
    """Creates an *async* insecure channel each call (cheap in aio)."""
    if GRPC_CHANNEL is None:
        grpc_host = os.getenv("ROBOT_GRPC_HOST", GRPC_HOST_DEFAULT)
        grpc_port = int(os.getenv("ROBOT_GRPC_PORT", str(GRPC_PORT_DEFAULT)))
        channel_target = f"{grpc_host}:{grpc_port}"
        logger.info(f"Creating gRPC channel to {channel_target}")
        GRPC_CHANNEL = grpc.aio.insecure_channel(channel_target)
    return GRPC_CHANNEL


def get_grpc_control_stub() -> pb_grpc.RobotControlServiceStub:
    ch = get_channel()
    return pb_grpc.RobotControlServiceStub(ch)


def get_grpc_vision_stub() -> pb_grpc.CameraServiceStub:
    ch = get_channel()
    return pb_grpc.CameraServiceStub(ch)


def pb_to_dict(msg) -> Dict[str, Any]:
    return MessageToDict(
        msg,
        preserving_proto_field_name=True,
        # including_default_value_fields 更名
        always_print_fields_with_no_presence=True,
        use_integers_for_enums=True,
    )


# ────────────────────────────────────────────────────────────────────────────
#  fastmcp application instance
# ────────────────────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="The MCP server providing robot control service",
    instructions="""This server will help you control a real robot."""
)


# ────────────────────────────────────────────────────────────────────────────
#  fastmcp “tools”
# ────────────────────────────────────────────────────────────────────────────

def common_svr_handler(func: Callable[..., Any]) -> Callable[..., Any]:
    """Install exception catch common body for server handlers"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except grpc.aio.AioRpcError as e:
            logger.error(f"gRPC error in {func.__name__}: {e}")
            return {
                "error": f"gRPC error: {e.code()}",
                "message": e.details()
            }
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}")
            return {"error": "Internal server error", "details": str(e)}

    return wrapper


# TIPS
# - Uses the function name (add) as the tool name.
# - Uses the function’s docstring (Adds two integer numbers...) as the tool description.
# - Generates an input schema based on the function’s parameters and type annotations.
# - Handles parameter validation and error reporting.

# @mcp.tool
# async def move_to_position(
#         x: Annotated[float, Field(description="x-coordinate of target position (meters)")],
#         y: Annotated[float, Field(description="y-coordinate of target position (meters)")],
#         z: Annotated[float, Field(description="z-coordinate of target position (meters)")],
#         roll: Annotated[float, Field(description="Target roll angle (radians)")],
#         pitch: Annotated[float, Field(description="Target pitch angle (radians)")],
#         yaw: Annotated[float, Field(description="Target yaw angle (radians)")],
#         max_speed: Annotated[float, Field(description="Maximum speed limit during movement (m/s)")] = 0.0,
#         tolerance: Annotated[float, Field(description="Position tolerance (meters)")] = 0.0,
#         orientation_tolerance: Annotated[float, Field(description="Orientation tolerance (radians)")] = 0.0
# ) -> dict:

@mcp.tool
@common_svr_handler
async def move_to_position(
        x: float, y: float, z: float,
        roll: float, pitch: float, yaw: float,
        max_speed: float = 0.0,
        tolerance: float = 0.0,
        orientation_tolerance: float = 0.0
) -> dict:
    """Controls the robot to move to the specified position and orientation. \
where the length unit defaults to meters (parameters x, y, z) \
and the angle unit defaults to degrees (parameters roll, pitch, yaw).
    It returns a dictionary containing code, message, actual distance traveled, and the final state of the machine."""
    logger.info(f"Received move_to_position request: "
                f"x={x}, y={y}, z={z}, "
                f"yaw={yaw}, pitch={pitch}, roll={roll}, "
                f"max_speed={max_speed}, tolerance={tolerance}, "
                f"orientation_tolerance={orientation_tolerance}")
    if tolerance < MIN_DIST_TOLERANCE:
        tolerance = MIN_DIST_TOLERANCE
    if orientation_tolerance < MIN_ORIENT_TOLERANCE:
        orientation_tolerance = MIN_ORIENT_TOLERANCE
    if max_speed < MIN_SPEED:
        max_speed = MIN_SPEED

    req = pb.TargetPosition(
        position=pb.Position(x=x, y=y, z=z),
        orientation=pb.Orientation(yaw=yaw, pitch=pitch, roll=roll),
        max_speed=max_speed,
        tolerance=tolerance,
        orientation_tolerance=orientation_tolerance,
    )
    stub = get_grpc_control_stub()
    logger.info("Calling MoveToPosition gRPC method")

    # stream move states
    last_state = pb.RobotState()
    async for state in stub.StreamMoveToPosition(req):
        logger.debug(
            "Robot movement checks: \n"
            f"\tPosition: x={state.position.x}, y={state.position.y}, z={state.position.z}\n"
            f"\tIs Moving: {state.state_code == pb.RobotState.MOVING}\n"
            f"\tBattery: {state.battery_level:.1%}"
        )

        if state.warnings:
            logger.warning(f"Robot current warnings: {', '.join(state.warnings)}")

        last_state = state
        if state.state_code != pb.RobotState.MOVING:
            # stream ended
            break

    # resp: pb.ControlResponse = await stub.MoveToPosition(req, timeout=30)

    # Get current position
    resp: pb.ControlResponse = await stub.GetCurrentPosition(pb.Empty())
    # override code & message for move_to_position
    if last_state.state_code != pb.RobotState.IDLE:
        logger.warning(f"Invalid robot state after move_to_position: {last_state.state_code}")
        resp.code = pb.ControlResponse.EMERGENCY_STOP
        resp.message = "Warning:" + (";".join(last_state.warnings))
    else:
        resp.message = "Finish moving process."
    logger.info(f"Received MoveToPosition response: code={resp.code}, message={resp.message}")

    return pb_to_dict(resp)


@mcp.tool
@common_svr_handler
async def get_current_position() -> dict:
    """Get the coordinates of the current position of the robot.
    Return a dictionary consisting of the coordinates of the current world coordinate system where the robot is located, \
the robot's pose, power level, warning messages, and other data."""
    logger.info("Received get_current_position request")
    stub = get_grpc_control_stub()
    logger.info("Calling GetCurrentPosition gRPC method")
    resp: pb.ControlResponse = await stub.GetCurrentPosition(pb.Empty())
    state: pb.RobotState = resp.current_state
    logger.info(f"Received GetCurrentPosition response: "
                f"code={resp.code}, message={resp.message}, "
                f"position=({state.position.x}, {state.position.y}, {state.position.z}), "
                f"battery={state.battery_level}")

    return pb_to_dict(state)


@mcp.tool
@common_svr_handler
async def emergency_stop() -> dict:
    """Immediately halt all robot motion. Return error status (code, message, suggested actions, etc.)"""
    logger.info("Received emergency_stop request")
    stub = get_grpc_control_stub()
    logger.info("Calling EmergencyStop gRPC method")
    resp: pb.ControlResponse = await stub.EmergencyStop(pb.Empty())
    logger.info(f"EmergencyStop executed: {resp.message} (code: {resp.code})")
    return pb_to_dict(resp)


@mcp.tool
@common_svr_handler
# async def pick_up_object(object_name_hint: str, x_min: int, y_min: int, x_max: int, y_max: int) -> dict:
#     """Pick up the specific object in front of the robot. object_name_hint represents the name for the object. \
# x_min, y_min, x_max, y_max represent the pixel coordinates of the upper left and lower right corners of the object \
# in the image. e.g., pick_up_object("banana", 220, 150, 520, 610).\
#     It returns action result (code, message, etc.)"""
async def pick_up_object(object_name_hint: str) -> dict:
    """Pick up the specific object in front of the robot. e.g., pick_up_object("banana").\
    Return action result (code, message, etc.)"""
    logger.info("Received pick_up_object request")
    stub = get_grpc_control_stub()
    logger.info("Calling PickUpObject gRPC method")

    try:
        # detected_object_name, x_min, y_min, x_max, y_max = await get_xyxy_from_image()
        x_min, y_min, x_max, y_max = await get_xyxy_from_image()
    except Exception as e:
        logger.error("failed to get xxyy from image when calling pick_up_object")
        logger.error(f"details: {e}")
        # return {"error": "Internal server error", "details": str(e)}
        x_min = y_min = x_max = y_max = 0

    # logger.info(f"Detected object '{detected_object_name}' at ({x_min},{y_min})-({x_max},{y_max}) "
    #             f"when picking up '{object_name_hint}'")
    logger.info(f"Detected object '{object_name_hint}' at ({x_min},{y_min})-({x_max},{y_max})")
    x_min = y_min = x_max = y_max = 0

    req = pb.PickOrPlaceCmd(
        cmd=object_name_hint,
        x_min=int(x_min), y_min=int(y_min),
        x_max=int(x_max), y_max=int(y_max))

    # stream pick-up
    async for state in stub.StreamPickUpObject(req):
        logger.debug(
            "Robot gripper checks: \n"
            f"\tIs Arm Moving: {state.is_moving_arm}\n"
            f"\tTarget Gripper Holding: {state.is_using_gripper}"
        )

        if state.warnings:
            logger.warning(f"Robot current warnings: {', '.join(state.warnings)}")

        if not state.is_moving_arm and state.state_code != pb.RobotState.MOVING:
            # stream ended
            break

    # Get current position
    resp: pb.ControlResponse = await stub.GetCurrentPosition(pb.Empty())
    resp.message += "\nFinish gripper pick-up operation."
    logger.info(f"Received PickUpObject response: code={resp.code}, message={resp.message}")
    return pb_to_dict(resp)


@mcp.tool
@common_svr_handler
async def place_object(object_name_hint: str) -> dict:
    """Place the object grabbed by the robot. Return action result (code, message, etc.)"""
    logger.info("Received place_object request")
    stub = get_grpc_control_stub()
    logger.info("Calling PlaceObject gRPC method")
    req = pb.PickOrPlaceCmd(cmd=object_name_hint)

    # stream pick-up
    async for state in stub.StreamPlaceObject(req):
        logger.debug(
            "Robot gripper checks: \n"
            f"\tIs Arm Moving: {state.is_moving_arm}\n"
            f"\tTarget Gripper Holding: {state.is_using_gripper}"
        )

        if state.warnings:
            logger.warning(f"Robot current warnings: {', '.join(state.warnings)}")

        if not state.is_moving_arm:
            # stream ended
            break

    # Get current position
    resp: pb.ControlResponse = await stub.GetCurrentPosition(pb.Empty())
    resp.message += "\nFinish gripper place operation."
    logger.info(f"Received PlaceObject response: code={resp.code}, message={resp.message}")
    return pb_to_dict(resp)


# @mcp.tool
# @common_svr_handler
# async def move_relative(direction: str, distance: float) -> dict:
#     """Do relative moving. `direction` can be 'LEFT_FORWARD', 'LEFT_BACKWARD', 'RIGHT_FORWARD', 'RIGHT_BACKWARD',\
#     'FORWARD', or 'BACKWARD'. `distance` is meters. \
#     Return action result (code, message, etc.)"""
#     logger.info("Received move_relative request")
#     stub = get_grpc_control_stub()
#     logger.info("Calling MoveRelative gRPC method")
#
#     # stream move states
#     last_state = pb.RobotState()
#     direction_ = pb.RobotDirection()
#     match direction:
#         case "LEFT_FORWARD":
#             direction_.direction = pb.RobotDirection.FORWARD_LEFT
#         case "LEFT_BACKWARD":
#             direction_.direction = pb.RobotDirection.BACKWARD_LEFT
#         case "RIGHT_FORWARD":
#             direction_.direction = pb.RobotDirection.FORWARD_RIGHT
#         case "RIGHT_BACKWARD":
#             direction_.direction = pb.RobotDirection.BACKWARD_RIGHT
#         case "FORWARD":
#             direction_.direction = pb.RobotDirection.FORWARD
#         case "BACKWARD":
#             direction_.direction = pb.RobotDirection.BACKWARD
#         case _:
#             return {
#                 "code": pb.ControlResponse.INVALID_TARGET,
#                 "message": f"invalid argument '{direction}' for param direction. "
#                 "Use LEFT_FORWARD', 'LEFT_BACKWARD', 'RIGHT_FORWARD', 'RIGHT_BACKWARD', 'FORWARD', or 'BACKWARD'"
#             }
#     direction_.distance = distance
#     async for state in stub.StreamMove(direction_):
#         logger.debug(
#             "Robot movement checks: \n"
#             f"\tPosition: x={state.position.x}, y={state.position.y}, z={state.position.z}\n"
#             f"\tIs Moving: {state.state_code == pb.RobotState.MOVING}\n"
#             f"\tBattery: {state.battery_level:.1%}"
#         )
#
#         if state.warnings:
#             logger.warning(f"Robot current warnings: {', '.join(state.warnings)}")
#
#         last_state = state
#         if state.state_code != pb.RobotState.MOVING:
#             # stream ended
#             break
#
#     # Get current position
#     resp: pb.ControlResponse = await stub.GetCurrentPosition(pb.Empty())
#     # override code & message for move_to_position
#     if last_state.state_code != pb.RobotState.IDLE:
#         logger.warning(f"Invalid robot state after move_relative: {last_state.state_code}")
#         resp.code = pb.ControlResponse.EMERGENCY_STOP
#         resp.message = "Warning:" + (";".join(last_state.warnings))
#     else:
#         resp.message = "Finish moving process."
#     logger.info(f"Received MoveRelative response: code={resp.code}, message={resp.message}")
#
#     return pb_to_dict(resp)


# @mcp.tool
# @common_svr_handler
async def get_robot_camera_image(save_dir: str | None = None) -> dict:
    """Get the camera image data of the current robot. `save_dir` set to None means 'do not save image to disk'\
    The returned image (in `image_url` field) is in base64 format."""
    logger.info("Received get_robot_camera_image request")
    stub = get_grpc_vision_stub()
    logger.info("Calling GetImage gRPC method")

    if save_dir is not None and not os.path.exists(save_dir):
        logger.error(f"cannot find specific save directory: '{save_dir}'. Not save the image")
        save_dir = None

    resp: pb.ImageResponse = await stub.GetImage(pb.Empty())

    image_bytes = resp.full_image
    pil_image = Image.open(io.BytesIO(image_bytes))
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    # 先转换格式为 JPEG
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    fn = ""
    if save_dir is not None:
        # generate file name
        # characters = string.ascii_letters + string.digits
        # random_str = ''.join(random.choice(characters) fo
        # r _ in range(16))
        # filename = f"robot-camera-{random_str}.jpg"
        filename = f"robot-camera-{time.time_ns()}.jpg"
        fn = os.path.join(save_dir, filename)
        pil_image.save(fn, format="JPEG")
    # 再转换成 base64 编码，方便传递给大模型
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # # (H, W, C)
    # image_np = np.array(pil_image)
    #
    # depth_data = resp.depth_image.depth_data
    # depth_min = resp.depth_image.min_depth
    # depth_max = resp.depth_image.max_depth
    #
    # height = resp.metadata.height
    # width = resp.metadata.width
    # depth_array = np.array(depth_data).reshape((height, width))

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
            "file": fn,
            # "detail": "high"
        }
    }


async def get_xyxy_from_image() -> Tuple[int, int, int, int]:
    # result = await _vlm.ainvoke([
    #     SystemMessage(content=VLM_SYS_PROMPT),
    #     HumanMessage(content=[
    #         {"type": "text", "text": VLM_DETECT_PROMPT},
    #         await get_robot_camera_image()
    #     ])
    # ])
    #
    #
    # try:
    #     res_json = json.loads(result.text())
    #     res_pts = res_json["xyxy"]
    #     return res_json["object"], res_pts[0], res_pts[1], res_pts[2], res_pts[3]
    # except Exception as e:
    #     msg = f"failed to decode '{result.text()}' to JSON template " \
    #         "{'object': 'name','xyxy': [xmin, ymin, xmax, ymax]}'"
    #     logger.error(msg)
    #     logger.error(f"details: {e}")
    #     raise RuntimeError(msg)

    os.makedirs("tmp", exist_ok=True)
    resp = await get_robot_camera_image("tmp")

    model = YOLO(MODEL_PATH)
    results = model.predict(resp["image_url"]["file"])
    try:
        boxes = results[0].boxes.xyxy
        for box in boxes:
            x_min, y_min, x_max, y_max = box.tolist()
            print(f"左上角: ({x_min}, {y_min}), 右下角: ({x_max}, {y_max})")
            return x_min, y_min, x_max, y_max
    except Exception as e:
        logger.warning(f"YOLO failed to recognize xyxy in image: {e}")
        return 0, 0, 0, 0


# ────────────────────────────────────────────────────────────────────────────
#  CLI entry-point
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stdio", action="store_true")
    parser.add_argument("--http-host", default="0.0.0.0")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--grpc-host", default=GRPC_HOST_DEFAULT)
    parser.add_argument("--grpc-port", type=int, default=GRPC_PORT_DEFAULT)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]),
    p_args = parser.parse_args()

    logger.setLevel(p_args.log_level)

    os.environ["ROBOT_GRPC_HOST"] = p_args.grpc_host
    os.environ["ROBOT_GRPC_PORT"] = str(p_args.grpc_port)

    logger.info(f"Starting MCP server with configuration:")
    logger.info(f"  gRPC target: {p_args.grpc_host}:{p_args.grpc_port}")
    logger.info(f"  HTTP server: {p_args.http_host}:{p_args.http_port}")
    logger.info(f"  Log level: {p_args.log_level}")
    logger.info(f"  Stdio mode: {p_args.stdio}")

    try:
        if p_args.stdio:
            # The transport uses Python standard input/output (stdio) for a local MCP server
            logger.info("Running in stdio mode")
            mcp.run(transport="stdio")
        else:
            logger.info(f"Starting HTTP server on {p_args.http_host}:{p_args.http_port}")
            mcp.run(transport="streamable-http", host=p_args.http_host, port=p_args.http_port)
    except asyncio.exceptions.CancelledError:
        logger.warning("MCP server stopped due to cancellation")
        sys.exit(0)
