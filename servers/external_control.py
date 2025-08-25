from flask import Flask, request, jsonify
import threading
from datetime import datetime

from common import prompts, config
from py_agent.robot_agent import RobotAgent


app = Flask(__name__)


_agent = RobotAgent(config.llm, config.server_params, prompts.SYSTEM_PROMPT)
ws_thread = threading.Thread(target=_agent.start_websocket_thread, daemon=True)
ws_thread.start()
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


def main():
    
    app.run(host='0.0.0.0', port=17111, debug=True, use_reloader=False)
    _agent.stop()


if __name__ == '__main__':
    main()
