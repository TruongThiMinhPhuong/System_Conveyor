"""
Web Server Entry Point
Run this to start the web interface
"""

from web.app import app, socketio
from utils import Config

if __name__ == '__main__':
    print("=" * 60)
    print("🌐 AI Fruit Sorting System - Web Interface")
    print("=" * 60)
    print(f"\n🔗 Access the interface at:")
    print(f"   Local: http://localhost:5000")
    print(f"   Network: http://<raspberry-pi-ip>:5000")
    print(f"\n📝 Press Ctrl+C to stop the server\n")
    
    # Run with SocketIO
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True
    )
