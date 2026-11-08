from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_works():
    r = client.post("/predict", json={"age": 55, "bmi": 32, "glucose": 130})
    assert r.status_code == 200
    assert r.json()["high_risk"] in (0, 1)


def test_bad_input_is_rejected():
    r = client.post("/predict", json={"age": "x", "bmi": 32, "glucose": 130})
    assert r.status_code == 422