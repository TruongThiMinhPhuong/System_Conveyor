"""Web interface package for AI Fruit Sorting System"""

from .app import create_app, socketio

__all__ = ['create_app', 'socketio']
