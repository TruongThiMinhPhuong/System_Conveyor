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
    print(f"   🖥️  Localhost:    http://localhost:5001")
    print(f"   🌐 Local IP:     http://127.0.0.1:5001")
    print(f"\n💡 For network access, use your computer's IP address")
    print(f"📝 Press Ctrl+C to stop the server\n")
    
    # Run with SocketIO on port 5001
    socketio.run(
        app,
        host='0.0.0.0',
        port=5001,
        debug=False,
        allow_unsafe_werkzeug=True
    )
