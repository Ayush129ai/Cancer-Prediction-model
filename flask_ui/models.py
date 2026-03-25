from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import json

db = SQLAlchemy()

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=True)
    features_json = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.Integer, nullable=False)
    probability_no_cancer = db.Column(db.Float, nullable=False)
    probability_cancer = db.Column(db.Float, nullable=False)
    label = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def set_features(self, features_dict):
        self.features_json = json.dumps(features_dict)

    def get_features(self):
        return json.loads(self.features_json)
