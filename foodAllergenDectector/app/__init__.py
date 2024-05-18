# app/__init__.py

from flask import Flask


def create_app():
    app = Flask(__name__)

    # Load configuration
    app.config['SECRET_KEY'] = 'your_secret_key'

    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)

    return app
