import os
from fastapi.testclient import TestClient
from main import app

# Set the necessary environment variables for testing
os.environ['NEXT_PUBLIC_BACKEND_URL'] = 'http://testserver'
os.environ['NEXT_PUBLIC_TTR_API_KEY'] = 'test_api_key'
os.environ['TTR_API_KEY'] = 'test_api_key'

# Create a TestClient using the FastAPI application
client = TestClient(app)

def test_page_js_api_calls():
    # Simulate a POST request to /system_message endpoint
    response = client.post(
        "/system_message",
        headers={"Authorization": f"Bearer {os.environ['NEXT_PUBLIC_TTR_API_KEY']}"},
        json={"text": "test message", "sender": "user"}
    )
    assert response.status_code == 200
    assert "system_message" in response.json()

    # Simulate a POST request to /chat_stream endpoint
    response = client.post(
        "/chat_stream",
        headers={"Authorization": f"Bearer {os.environ['NEXT_PUBLIC_TTR_API_KEY']}"},
        json=[{"text": "test message", "sender": "user"}]
    )
    assert response.status_code == 200
    # The response for /chat_stream is a streaming response, so we won't check its content here

    # Add more tests as needed for other endpoints that page.js interacts with

# Run the tests
if __name__ == "__main__":
    test_page_js_api_calls()
    print("All tests passed!")
