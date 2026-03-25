import os
import sys
import time
import subprocess
import pytest
from playwright.sync_api import Page, expect

# We use a fixture to start the Flask server in the background
@pytest.fixture(scope="module", autouse=True)
def start_flask_app():
    # Setup environment
    env = os.environ.copy()
    # Path to flask_ui/app.py
    app_path = os.path.join(os.path.dirname(__file__), '..', 'flask_ui', 'app.py')
    
    # Run the flask app
    process = subprocess.Popen(
        [sys.executable, app_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for the server to start
    time.sleep(5)
    
    yield
    
    # Teardown the server
    process.terminate()
    process.wait()

def test_homepage_loads(page: Page):
    page.goto("http://127.0.0.1:5000/")
    expect(page).to_have_title("Cancer Prediction Model")
    expect(page.locator("h1")).to_contain_text("Breast Cancer Prediction UI")

def test_prediction_flow(page: Page):
    page.goto("http://127.0.0.1:5000/")
    
    # Ensure form exists
    form = page.locator("#prediction-form")
    expect(form).to_be_visible()
    
    # Fill in the optional User ID
    page.fill("input#user_id", "Playwright Test User")
    
    # Click predict
    page.click("button#predict-btn")
    
    # Expect success message (the alert that flashed)
    success_alert = page.locator(".alert-success")
    expect(success_alert).to_be_visible()
    expect(success_alert).to_contain_text("Prediction successful!")
    
    # Expect history table to be populated
    history_table = page.locator("#history-table tbody tr")
    expect(history_table.first).to_contain_text("Playwright Test User")
    expect(history_table.first).to_contain_text("%") # Probability cells
