import pytest
from playwright.sync_api import Page, expect
import subprocess
import time
import requests

@pytest.fixture(scope="session", autouse=True)
def start_server():
    """Start the Flask server before tests run and shut it down after."""
    import sys
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # Start flask app
    f = open("flask_error.log", "w")
    process = subprocess.Popen(
        ["C:/Python314/python.exe", "-m", "flask", "--app", "flask_ui/app.py", "run", "--port", "5001"],
        env=env,
        stderr=f,
        stdout=f
    )
    
    # Wait for server to start
    for _ in range(30):
        try:
            response = requests.get("http://localhost:5001/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    
    yield
    
    # Teardown
    process.terminate()
    f.close()

def test_home_page(page: Page):
    page.goto("http://localhost:5001/")
    expect(page).to_have_title("Cancer Prediction Model")
    # Verify the presence of form or elements
    page.wait_for_selector("form")

def test_health_endpoint():
    response = requests.get("http://localhost:5001/health")
    assert response.status_code == 200
    assert response.json()["status"] == 'healthy'

def test_ready_endpoint():
    response = requests.get("http://localhost:5001/ready")
    assert response.status_code == 200
    assert response.json()["status"] == 'ready'

def test_metrics_endpoint():
    response = requests.get("http://localhost:5001/metrics")
    assert response.status_code == 200
    assert "flask_http_request_duration_seconds" in response.text