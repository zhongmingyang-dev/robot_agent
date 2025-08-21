from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Position(_message.Message):
    __slots__ = ("x", "y", "z")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

class Orientation(_message.Message):
    __slots__ = ("yaw", "pitch", "roll")
    YAW_FIELD_NUMBER: _ClassVar[int]
    PITCH_FIELD_NUMBER: _ClassVar[int]
    ROLL_FIELD_NUMBER: _ClassVar[int]
    yaw: float
    pitch: float
    roll: float
    def __init__(self, yaw: _Optional[float] = ..., pitch: _Optional[float] = ..., roll: _Optional[float] = ...) -> None: ...

class TargetPosition(_message.Message):
    __slots__ = ("position", "orientation", "max_speed", "tolerance", "orientation_tolerance")
    POSITION_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    MAX_SPEED_FIELD_NUMBER: _ClassVar[int]
    TOLERANCE_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_TOLERANCE_FIELD_NUMBER: _ClassVar[int]
    position: Position
    orientation: Orientation
    max_speed: float
    tolerance: float
    orientation_tolerance: float
    def __init__(self, position: _Optional[_Union[Position, _Mapping]] = ..., orientation: _Optional[_Union[Orientation, _Mapping]] = ..., max_speed: _Optional[float] = ..., tolerance: _Optional[float] = ..., orientation_tolerance: _Optional[float] = ...) -> None: ...

class RobotState(_message.Message):
    __slots__ = ("position", "orientation", "battery_level", "warnings", "state_code", "is_using_gripper", "is_moving_arm", "timestamp")
    class StateEnum(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        IDLE: _ClassVar[RobotState.StateEnum]
        MOVING: _ClassVar[RobotState.StateEnum]
        EMERGE_STOP: _ClassVar[RobotState.StateEnum]
    IDLE: RobotState.StateEnum
    MOVING: RobotState.StateEnum
    EMERGE_STOP: RobotState.StateEnum
    POSITION_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    BATTERY_LEVEL_FIELD_NUMBER: _ClassVar[int]
    WARNINGS_FIELD_NUMBER: _ClassVar[int]
    STATE_CODE_FIELD_NUMBER: _ClassVar[int]
    IS_USING_GRIPPER_FIELD_NUMBER: _ClassVar[int]
    IS_MOVING_ARM_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    position: Position
    orientation: Orientation
    battery_level: float
    warnings: _containers.RepeatedScalarFieldContainer[str]
    state_code: RobotState.StateEnum
    is_using_gripper: bool
    is_moving_arm: bool
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, position: _Optional[_Union[Position, _Mapping]] = ..., orientation: _Optional[_Union[Orientation, _Mapping]] = ..., battery_level: _Optional[float] = ..., warnings: _Optional[_Iterable[str]] = ..., state_code: _Optional[_Union[RobotState.StateEnum, str]] = ..., is_using_gripper: bool = ..., is_moving_arm: bool = ..., timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class ControlResponse(_message.Message):
    __slots__ = ("code", "message", "current_state", "suggested_actions")
    class ResultCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SUCCESS: _ClassVar[ControlResponse.ResultCode]
        PARTIAL_SUCCESS: _ClassVar[ControlResponse.ResultCode]
        OBSTACLE_DETECTED: _ClassVar[ControlResponse.ResultCode]
        LOW_BATTERY: _ClassVar[ControlResponse.ResultCode]
        OUT_OF_RANGE: _ClassVar[ControlResponse.ResultCode]
        TIMEOUT: _ClassVar[ControlResponse.ResultCode]
        EMERGENCY_STOP: _ClassVar[ControlResponse.ResultCode]
        INVALID_TARGET: _ClassVar[ControlResponse.ResultCode]
    SUCCESS: ControlResponse.ResultCode
    PARTIAL_SUCCESS: ControlResponse.ResultCode
    OBSTACLE_DETECTED: ControlResponse.ResultCode
    LOW_BATTERY: ControlResponse.ResultCode
    OUT_OF_RANGE: ControlResponse.ResultCode
    TIMEOUT: ControlResponse.ResultCode
    EMERGENCY_STOP: ControlResponse.ResultCode
    INVALID_TARGET: ControlResponse.ResultCode
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STATE_FIELD_NUMBER: _ClassVar[int]
    SUGGESTED_ACTIONS_FIELD_NUMBER: _ClassVar[int]
    code: ControlResponse.ResultCode
    message: str
    current_state: RobotState
    suggested_actions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, code: _Optional[_Union[ControlResponse.ResultCode, str]] = ..., message: _Optional[str] = ..., current_state: _Optional[_Union[RobotState, _Mapping]] = ..., suggested_actions: _Optional[_Iterable[str]] = ...) -> None: ...

class RobotDirection(_message.Message):
    __slots__ = ("direction", "distance")
    class Direction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        FORWARD: _ClassVar[RobotDirection.Direction]
        BACKWARD: _ClassVar[RobotDirection.Direction]
        FORWARD_LEFT: _ClassVar[RobotDirection.Direction]
        FORWARD_RIGHT: _ClassVar[RobotDirection.Direction]
        BACKWARD_LEFT: _ClassVar[RobotDirection.Direction]
        BACKWARD_RIGHT: _ClassVar[RobotDirection.Direction]
    FORWARD: RobotDirection.Direction
    BACKWARD: RobotDirection.Direction
    FORWARD_LEFT: RobotDirection.Direction
    FORWARD_RIGHT: RobotDirection.Direction
    BACKWARD_LEFT: RobotDirection.Direction
    BACKWARD_RIGHT: RobotDirection.Direction
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    direction: RobotDirection.Direction
    distance: int
    def __init__(self, direction: _Optional[_Union[RobotDirection.Direction, str]] = ..., distance: _Optional[int] = ...) -> None: ...

class PickOrPlaceCmd(_message.Message):
    __slots__ = ("cmd", "x_min", "y_min", "x_max", "y_max")
    CMD_FIELD_NUMBER: _ClassVar[int]
    X_MIN_FIELD_NUMBER: _ClassVar[int]
    Y_MIN_FIELD_NUMBER: _ClassVar[int]
    X_MAX_FIELD_NUMBER: _ClassVar[int]
    Y_MAX_FIELD_NUMBER: _ClassVar[int]
    cmd: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    def __init__(self, cmd: _Optional[str] = ..., x_min: _Optional[int] = ..., y_min: _Optional[int] = ..., x_max: _Optional[int] = ..., y_max: _Optional[int] = ...) -> None: ...

class CameraConfig(_message.Message):
    __slots__ = ("format", "resolution", "frame_rate", "auto_exposure", "exposure_time", "gain", "color", "quality", "enable_depth", "enable_pointcloud")
    class ImageFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNKNOWN_FORMAT: _ClassVar[CameraConfig.ImageFormat]
        JPEG: _ClassVar[CameraConfig.ImageFormat]
        PNG: _ClassVar[CameraConfig.ImageFormat]
        RAW: _ClassVar[CameraConfig.ImageFormat]
        H264_FRAME: _ClassVar[CameraConfig.ImageFormat]
    UNKNOWN_FORMAT: CameraConfig.ImageFormat
    JPEG: CameraConfig.ImageFormat
    PNG: CameraConfig.ImageFormat
    RAW: CameraConfig.ImageFormat
    H264_FRAME: CameraConfig.ImageFormat
    class Resolution(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        RES_UNKNOWN: _ClassVar[CameraConfig.Resolution]
        RES_640x480: _ClassVar[CameraConfig.Resolution]
        RES_1280x720: _ClassVar[CameraConfig.Resolution]
        RES_1920x1080: _ClassVar[CameraConfig.Resolution]
        RES_3840x2160: _ClassVar[CameraConfig.Resolution]
    RES_UNKNOWN: CameraConfig.Resolution
    RES_640x480: CameraConfig.Resolution
    RES_1280x720: CameraConfig.Resolution
    RES_1920x1080: CameraConfig.Resolution
    RES_3840x2160: CameraConfig.Resolution
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    RESOLUTION_FIELD_NUMBER: _ClassVar[int]
    FRAME_RATE_FIELD_NUMBER: _ClassVar[int]
    AUTO_EXPOSURE_FIELD_NUMBER: _ClassVar[int]
    EXPOSURE_TIME_FIELD_NUMBER: _ClassVar[int]
    GAIN_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    ENABLE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    ENABLE_POINTCLOUD_FIELD_NUMBER: _ClassVar[int]
    format: CameraConfig.ImageFormat
    resolution: CameraConfig.Resolution
    frame_rate: float
    auto_exposure: bool
    exposure_time: float
    gain: float
    color: bool
    quality: float
    enable_depth: bool
    enable_pointcloud: bool
    def __init__(self, format: _Optional[_Union[CameraConfig.ImageFormat, str]] = ..., resolution: _Optional[_Union[CameraConfig.Resolution, str]] = ..., frame_rate: _Optional[float] = ..., auto_exposure: bool = ..., exposure_time: _Optional[float] = ..., gain: _Optional[float] = ..., color: bool = ..., quality: _Optional[float] = ..., enable_depth: bool = ..., enable_pointcloud: bool = ...) -> None: ...

class ImageMetadata(_message.Message):
    __slots__ = ("capture_time", "width", "height", "channels", "camera_position", "camera_orientation", "custom_metadata")
    class CustomMetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CAPTURE_TIME_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    CAMERA_POSITION_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_METADATA_FIELD_NUMBER: _ClassVar[int]
    capture_time: _timestamp_pb2.Timestamp
    width: int
    height: int
    channels: int
    camera_position: Position
    camera_orientation: Orientation
    custom_metadata: _containers.ScalarMap[str, str]
    def __init__(self, capture_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., channels: _Optional[int] = ..., camera_position: _Optional[_Union[Position, _Mapping]] = ..., camera_orientation: _Optional[_Union[Orientation, _Mapping]] = ..., custom_metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class DepthImage(_message.Message):
    __slots__ = ("depth_data", "min_depth", "max_depth")
    DEPTH_DATA_FIELD_NUMBER: _ClassVar[int]
    MIN_DEPTH_FIELD_NUMBER: _ClassVar[int]
    MAX_DEPTH_FIELD_NUMBER: _ClassVar[int]
    depth_data: _containers.RepeatedScalarFieldContainer[float]
    min_depth: float
    max_depth: float
    def __init__(self, depth_data: _Optional[_Iterable[float]] = ..., min_depth: _Optional[float] = ..., max_depth: _Optional[float] = ...) -> None: ...

class ImageResponse(_message.Message):
    __slots__ = ("metadata", "config", "full_image", "depth_image")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    FULL_IMAGE_FIELD_NUMBER: _ClassVar[int]
    DEPTH_IMAGE_FIELD_NUMBER: _ClassVar[int]
    metadata: ImageMetadata
    config: CameraConfig
    full_image: bytes
    depth_image: DepthImage
    def __init__(self, metadata: _Optional[_Union[ImageMetadata, _Mapping]] = ..., config: _Optional[_Union[CameraConfig, _Mapping]] = ..., full_image: _Optional[bytes] = ..., depth_image: _Optional[_Union[DepthImage, _Mapping]] = ...) -> None: ...
