import starlette.testclient
import pytest
import api.app

@pytest.fixture
def client () -> starlette.testclient.TestClient:
    return starlette.testclient.TestClient(api.app.app)