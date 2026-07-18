"""WSGI entry point for the QuantFlow WebApp."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webapp.app import create_app
from webapp.services.auth_service import User
from webapp.config import Config

app = create_app()

with app.app_context():
    User.create_db_tables(Config.DATABASE_PATH)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
