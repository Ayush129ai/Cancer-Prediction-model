import os

try:
    from supabase import create_client
except ImportError:
    create_client = None

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

client = None

if create_client and SUPABASE_URL and SUPABASE_KEY:
    client = create_client(SUPABASE_URL, SUPABASE_KEY)


def log_prediction(user_id: str, features: dict, prediction: int, probability: float):
    """Log a prediction event to Supabase."""
    if client is None:
        return None

    payload = {
        'user_id': user_id,
        'features': features,
        'prediction': prediction,
        'probability': probability
    }
    return client.table('predictions').insert(payload).execute()
