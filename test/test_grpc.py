import grpc
import time
import random
import argparse
import proto.robot_control_pb2 as rc
import proto.robot_control_pb2_grpc as rc_grpc


def run_simple_move(stub):
    """测试简单移动接口"""
    print("\n=== 测试简单移动接口 (MoveToPosition) ===")

    # 创建随机目标位置
    target = rc.TargetPosition(
        position=rc.Position(
            x=random.uniform(-10, 10),
            y=random.uniform(-10, 10),
            z=0
        ),
        max_speed=random.uniform(0.2, 1.0),
        tolerance=0.1
    )

    print(f"发送移动请求: 目标位置 ({target.position.x:.2f}, {target.position.y:.2f}), "
          f"速度 {target.max_speed:.2f}m/s")

    start_time = time.time()
    response = stub.MoveToPosition(target)

    print(f"移动完成! 耗时: {time.time() - start_time:.2f}秒")
    print(f"结果代码: {rc.MoveResult.ResultCode.Name(response.code)}")
    print(f"消息: {response.message}")

    if response.final_state:
        state = response.final_state
        print(f"最终位置: ({state.position.x:.2f}, {state.position.y:.2f}, {state.position.z:.2f})")
        print(f"电池电量: {state.battery_level:.1%}")
        print(f"移动状态: {'移动中' if state.is_moving else '已停止'}")
        print(f"警告: {state.warnings if state.warnings else '无'}")
        print(f"时间戳: {state.timestamp.ToDatetime().strftime('%Y-%m-%d %H:%M:%S')}")


def run_stream_move(stub):
    """测试流式移动接口"""
    print("\n=== 测试流式移动接口 (StreamMoveToPosition) ===")

    # 创建随机目标位置
    target = rc.TargetPosition(
        position=rc.Position(
            x=random.uniform(5, 15),
            y=random.uniform(5, 15),
            z=0
        ),
        max_speed=random.uniform(0.3, 0.8),
        tolerance=0.1
    )

    print(f"发送移动请求: 目标位置 ({target.position.x:.2f}, {target.position.y:.2f}), "
          f"速度 {target.max_speed:.2f}m/s")
    print("接收实时状态更新... (按Ctrl+C停止)")

    start_time = time.time()
    state_count = 0
    last_position = (0, 0, 0)

    try:
        for state in stub.StreamMoveToPosition(target):
            state_count += 1
            dist = ((state.position.x - last_position[0]) ** 2 +
                    (state.position.y - last_position[1]) ** 2) ** 0.5
            last_position = (state.position.x, state.position.y, state.position.z)

            print(f"\n状态 #{state_count} [耗时: {time.time() - start_time:.1f}s]")
            print(f"位置: ({state.position.x:.2f}, {state.position.y:.2f}) "
                  f"(移动: {dist:.3f}m)")
            print(f"电池: {state.battery_level:.1%} | 移动状态: {'是' if state.is_moving else '否'}")

            if state.warnings:
                print(f"! 警告: {', '.join(state.warnings)}")

            # 模拟处理延迟
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n用户中断了流式传输")

    print(f"\n移动结束! 共接收 {state_count} 个状态更新")
    print(f"最终位置: ({last_position[0]:.2f}, {last_position[1]:.2f})")
    print(f"总耗时: {time.time() - start_time:.1f}秒")


def run_advanced_move(stub):
    """测试高级移动控制接口"""
    print("\n=== 测试高级移动控制 (AdvancedMoveControl) ===")
    print("使用双向流控制机器人 (输入命令控制机器人)")
    print("可用命令: target, speed, pause, resume, cancel, status, quit")

    # 创建双向流
    response_stream = stub.AdvancedMoveControl(generate_commands())

    try:
        # 处理服务器响应
        for state in response_stream:
            print(f"\n机器人状态 [时间: {state.timestamp.ToDatetime().strftime('%H:%M:%S')}]")
            print(f"位置: ({state.position.x:.2f}, {state.position.y:.2f}) | "
                  f"电池: {state.battery_level:.1%} | 移动: {'是' if state.is_moving else '否'}")

            if state.warnings:
                print(f"! 警告: {', '.join(state.warnings)}")

            # 添加短暂延迟以显示状态
            time.sleep(0.5)
    except grpc.RpcError as e:
        print(f"RPC错误: {e.details()}")


def generate_commands():
    """生成高级移动控制命令"""
    commands = [
        "target",  # 设置初始目标
        "speed",  # 调整速度
        "pause",  # 暂停
        "resume",  # 恢复
        "status",  # 获取状态
        "speed",  # 再次调整速度
        "target",  # 设置新目标
        "cancel"  # 取消移动
    ]

    # 初始目标
    yield create_command("target", rc.TargetPosition(
        position=rc.Position(x=8, y=8, z=0),
        max_speed=0.4,
        tolerance=0.1
    ))
    time.sleep(2)  # 等待机器人开始移动

    for cmd in commands:
        user_input = input(f"\n输入命令 [{cmd}]: ").strip().lower() or cmd

        if user_input == "quit":
            break

        if user_input == "target":
            # 创建新目标
            new_target = rc.TargetPosition(
                position=rc.Position(
                    x=random.uniform(-5, 15),
                    y=random.uniform(-5, 15),
                    z=0
                ),
                max_speed=random.uniform(0.3, 1.0),
                tolerance=0.1
            )
            yield create_command("target", new_target)
            print(f"设置新目标: ({new_target.position.x:.2f}, {new_target.position.y:.2f}), "
                  f"速度 {new_target.max_speed:.2f}m/s")

        elif user_input == "speed":
            # 调整速度
            new_speed = random.uniform(0.2, 1.0)
            yield create_command("speed", new_speed)
            print(f"调整速度至: {new_speed:.2f}m/s")

        elif user_input == "pause":
            yield create_command("pause", True)
            print("暂停移动")

        elif user_input == "resume":
            yield create_command("resume", True)
            print("恢复移动")

        elif user_input == "cancel":
            yield create_command("cancel", True)
            print("取消移动")
            break

        elif user_input == "status":
            yield create_command("get_status", None)
            print("请求状态更新")

        else:
            print(f"未知命令: {user_input}")

        # 命令之间延迟
        time.sleep(1.5)

    print("高级控制会话结束")


def create_command(cmd_type, value):
    """创建MoveCommand对象"""
    cmd = rc.MoveCommand()

    if cmd_type == "target":
        cmd.set_target.CopyFrom(value)
    elif cmd_type == "speed":
        cmd.adjust_speed = value
    elif cmd_type == "pause":
        cmd.pause = True
    elif cmd_type == "resume":
        cmd.resume = True
    elif cmd_type == "cancel":
        cmd.cancel = True
    elif cmd_type == "get_status":
        cmd.get_status.CopyFrom(rc.Empty())

    return cmd


def run_current_position(stub):
    """测试获取当前位置"""
    print("\n=== 测试获取当前位置 (GetCurrentPosition) ===")

    response = stub.GetCurrentPosition(rc.Empty())

    print("当前机器人状态:")
    print(f"位置: ({response.position.x:.2f}, {response.position.y:.2f}, {response.position.z:.2f})")
    print(f"电池: {response.battery_level:.1%}")
    print(f"移动状态: {'是' if response.is_moving else '否'}")
    print(f"警告: {response.warnings if response.warnings else '无'}")
    print(f"时间戳: {response.timestamp.ToDatetime().strftime('%Y-%m-%d %H:%M:%S')}")


def run_emergency_stop(stub):
    """测试紧急停止"""
    print("\n=== 测试紧急停止 (EmergencyStop) ===")

    confirm = input("确定要触发紧急停止吗? (y/n): ").strip().lower()
    if confirm != 'y':
        print("取消紧急停止")
        return

    print("发送紧急停止请求...")
    response = stub.EmergencyStop(rc.Empty())

    print("\n紧急停止结果:")
    print(f"错误代码: {response.error_code}")
    print(f"消息: {response.error_message}")
    print(f"建议操作: {', '.join(response.suggested_actions)}")

    # 获取停止后状态
    print("\n停止后状态:")
    state = stub.GetCurrentPosition(rc.Empty())
    print(f"位置: ({state.position.x:.2f}, {state.position.y:.2f})")
    print(f"移动状态: {'是' if state.is_moving else '否'}")
    print(f"警告: {state.warnings}")


def run_test_sequence(stub):
    """运行完整测试序列"""
    tests = [
        ("简单移动测试", run_simple_move),
        ("流式移动测试", run_stream_move),
        ("当前位置测试", run_current_position),
        ("高级控制测试", run_advanced_move),
        ("紧急停止测试", run_emergency_stop)
    ]

    for name, test_func in tests:
        print(f"\n{'=' * 40}")
        print(f" 开始测试: {name} ")
        print('=' * 40)
        test_func(stub)
        time.sleep(1)  # 测试之间短暂暂停


def main():
    parser = argparse.ArgumentParser(description='机器人控制 gRPC 客户端')
    parser.add_argument('--host', type=str, default='localhost', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=50051, help='服务器端口号')
    parser.add_argument('--test', type=str, choices=[
        'simple', 'stream', 'advanced', 'position', 'emergency', 'all'
    ], default='all', help='选择测试类型')

    args = parser.parse_args()

    # 设置 gRPC 连接
    channel = grpc.insecure_channel(f'{args.host}:{args.port}')
    stub = rc_grpc.RobotControlServiceStub(channel)

    print(f"连接到机器人控制服务: {args.host}:{args.port}")

    try:
        # 测试服务器连接
        stub.GetCurrentPosition(rc.Empty())
        print("服务器连接成功!")
    except grpc.RpcError as e:
        print(f"连接服务器失败: {e.details()}")
        return

    # 执行选择的测试
    if args.test == 'all':
        run_test_sequence(stub)
    elif args.test == 'simple':
        run_simple_move(stub)
    elif args.test == 'stream':
        run_stream_move(stub)
    elif args.test == 'advanced':
        run_advanced_move(stub)
    elif args.test == 'position':
        run_current_position(stub)
    elif args.test == 'emergency':
        run_emergency_stop(stub)


if __name__ == '__main__':
    main()
