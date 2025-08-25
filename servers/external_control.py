from flask import Flask, request, jsonify
import threading
from datetime import datetime

from common import prompts, config
from py_agent.robot_agent import RobotAgent
import asyncio
import websockets

app = Flask(__name__)

_agent = RobotAgent(config.llm, config.server_params, prompts.SYSTEM_PROMPT)
_agent.start()


@app.route('/api/delivery/notification', methods=['POST'])
def handle_notification():
    # 获取 JSON 数据
    data = request.json

    # 验证必需字段
    required_fields = ['platform', 'location', 'phone', 'image_url', 'timestamp', 'original_sms']
    if not all(field in data for field in required_fields):
        return jsonify({"error": f"Missing required fields: {','.join(required_fields)}"}), 400

    try:
        # 打印接收到的数据（实际使用时可以替换为数据库存储或其他处理）
        print("\n" + "=" * 50)
        print(f"[{datetime.now().isoformat()}] 收到外卖通知:")
        print(f"平台: {data['platform']}")
        print(f"位置: {data['location']}")
        print(f"电话: {data['phone']}")
        print(f"图片URL: {data['image_url']}")
        print(f"时间戳: {data['timestamp']}")
        print(f"原始短信: {data['original_sms']}")
        print("=" * 50)

        # 这里可以添加其他处理逻辑，如：
        # 1. 保存到数据库
        # 2. 发送给其他服务
        # 3. 触发通知等

        _agent.submit_message(prompts.MSG + data['original_sms'])

        return jsonify({
            "status": "success",
            "message": "Notification processed.",
            # "received_data": data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

connected_clients = set()

async def register(websocket):
    connected_clients.add(websocket)
    print(f"✅ 客户端接入: {len(connected_clients)} 个在线")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.remove(websocket)
        print(f"❌ 客户端断开: {len(connected_clients)} 个在线")

async def handle_client(self,websocket):
    # 注册客户端
    await self.register(websocket)

    # 这里处理客户端主动发来的消息
    async for message in websocket:
        print(f"📩 收到客户端消息: {message}")

async def send_to_user(msg: str, websocket=None):
    """
    调用 Agent 并把结果推送给指定 websocket，
    如果 websocket=None，则广播给所有客户端
    """

    if websocket:
        targets = [websocket]
    else:
        targets = list(connected_clients)

    if not targets:
        print("⚠️ 没有客户端在线，消息不会被发送")
        return

    for ws in targets:
        try:
            await ws.send(msg)
        except Exception as e:
            print(f"❌ 向客户端发送失败: {e}")

async def websocket_service():
    async with websockets.serve(handle_client, "0.0.0.0", 9000):
        print("🚀 WebSocket 服务已启动: ws://0.0.0.0:9000")
        await asyncio.Future()  # 永不退出

def start_websocket_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_service())

def main():
    ws_thread = threading.Thread(target=start_websocket_thread, daemon=True)
    ws_thread.start()
    
    app.run(host='0.0.0.0', port=17111, debug=True, use_reloader=False)
    _agent.stop()


if __name__ == '__main__':
    main()
