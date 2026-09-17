"""Validation suite for the stateful order & export API.

The service under test is the vendored Express mock at
``examples/express_mock/server.js`` — a behavior-identical copy of the public
reference server at
https://gist.github.com/sharanya-lb/b429b6e807f95a8df8216c4343ff6766
(only a /healthz readiness endpoint and a PORT env override were added).
It is started automatically by the session fixture; tests are named purely
for the stateful API they validate.

Tests validate both the live behavior AND the spec file: responses are
checked against the hand-written OpenAPI 3.1 contract at
``examples/express_mock/openapi.yaml`` via the Layer-1 ContractValidator.

Layer-2 policy: cassettes are only used for the stateless auth route. The
order lifecycle and export jobs are time-driven server state, so those tests
always run live against the running service.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from examples.tests.conftest import poll_until

BEARER = "Bearer mock-jwt-bearer-token-12345"
CORRELATION = "X-Correlation-ID"
ORDER_ID_RE = re.compile(r"^ORD-\d{5}$")
JOB_ID_RE = re.compile(r"^JOB-\d{5}$")


def _headers(token: str = BEARER, correlation: str = "driftgate-test") -> dict[str, str]:
    headers = {"Authorization": token}
    if correlation:
        headers[CORRELATION] = correlation
    return headers


def _order_payload(customer_id: str = "cust-42") -> dict[str, object]:
    return {
        "customerId": customer_id,
        "items": [
            {"sku": "SKU-1", "quantity": 2, "unitPrice": 25.0},
            {"sku": "SKU-2", "quantity": 4, "unitPrice": 10.0},
        ],
        "shippingAddress": "1 Test Way",
    }


def _create_order(client) -> dict[str, object]:
    response = client.post(
        "/orders",
        json=_order_payload(),
        headers=_headers(correlation="driftgate-create"),
    )
    assert response.status_code == 202
    return response.json()


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


class TestAuth:
    def test_login_success_shape(self, express_client) -> None:
        response = express_client.post("/auth/login", json={"username": "drift", "apiKey": "gate"})
        assert response.status_code == 200
        body = response.json()
        assert body["token"] == "mock-jwt-bearer-token-12345"
        assert body["expiresIn"] == 3600
        assert (
            express_client.contracts["login"].validate_response(
                "POST", "/v1/auth/login", 200, body=body
            )
            == []
        )

    def test_login_missing_username(self, express_client) -> None:
        response = express_client.post("/auth/login", json={"apiKey": "gate"})
        assert response.status_code == 400
        assert response.json() == {"error": "Missing credentials"}

    def test_login_missing_api_key(self, express_client) -> None:
        response = express_client.post("/auth/login", json={"username": "drift"})
        assert response.status_code == 400
        assert response.json() == {"error": "Missing credentials"}

    def test_login_missing_credentials(self, express_client) -> None:
        response = express_client.post("/auth/login", json={})
        assert response.status_code == 400
        assert response.json() == {"error": "Missing credentials"}

    def test_protected_route_rejects_missing_token(self, express_client) -> None:
        response = express_client.post("/orders", json=_order_payload(), headers={})
        assert response.status_code == 401
        assert "token" in response.json()["error"].casefold()


# --------------------------------------------------------------------------
# Order creation
# --------------------------------------------------------------------------


class TestOrderCreation:
    def test_create_order_shape_and_math(self, express_client) -> None:
        response = express_client.post("/orders", json=_order_payload(), headers=_headers())
        assert response.status_code == 202
        body = response.json()
        assert ORDER_ID_RE.fullmatch(body["orderId"])
        assert body["status"] == "PENDING"
        assert body["totalAmount"] == 25.0 * 2 + 10.0 * 4
        parsed = datetime.fromisoformat(body["createdAt"].replace("Z", "+00:00"))
        assert parsed.tzinfo is UTC
        result = express_client.contracts["order_create"].validate_response(
            "POST", "/v1/orders", 202, body=body
        )
        assert result == []

    def test_create_order_missing_correlation_header(self, express_client) -> None:
        headers = _headers()
        del headers[CORRELATION]
        response = express_client.post("/orders", json=_order_payload(), headers=headers)
        assert response.status_code == 400
        assert "Correlation" in response.json()["error"]

    def test_create_order_empty_items(self, express_client) -> None:
        payload = _order_payload()
        payload["items"] = []
        response = express_client.post("/orders", json=payload, headers=_headers())
        assert response.status_code == 400
        assert response.json() == {"error": "Invalid payload"}

    def test_create_order_missing_fields(self, express_client) -> None:
        payload = _order_payload()
        del payload["shippingAddress"]
        response = express_client.post("/orders", json=payload, headers=_headers())
        assert response.status_code == 400
        assert response.json() == {"error": "Invalid payload"}

    def test_create_order_unauthorized(self, express_client) -> None:
        response = express_client.post("/orders", json=_order_payload(), headers={CORRELATION: "x"})
        assert response.status_code == 401


# --------------------------------------------------------------------------
# Order retrieval + state machine (slow: real 5s/15s transitions)
# --------------------------------------------------------------------------


class TestOrderRead:
    def test_read_back_new_order(self, express_client) -> None:
        created = express_client.post("/orders", json=_order_payload(), headers=_headers()).json()
        response = express_client.get(f"/orders/{created['orderId']}", headers=_headers())
        assert response.status_code == 200
        body = response.json()
        assert body["orderId"] == created["orderId"]
        assert body["customerId"] == _order_payload()["customerId"]
        assert body["status"] == "PENDING"

    def test_read_unknown_order(self, express_client) -> None:
        response = express_client.get("/orders/ORD-99999", headers=_headers())
        assert response.status_code == 404
        assert response.json() == {"error": "Order not found"}

    @pytest.mark.slow
    def test_state_transitions_processing_then_completed(self, express_client) -> None:

        created = express_client.post("/orders", json=_order_payload(), headers=_headers()).json()
        order_id = created["orderId"]

        processing = poll_until(
            lambda: express_client.get(f"/orders/{order_id}", headers=_headers()).json(),
            lambda body: body["status"] == "PROCESSING",
            interval_s=0.5,
            timeout_s=10.0,
        )
        assert processing["status"] == "PROCESSING"

        completed = poll_until(
            lambda: express_client.get(f"/orders/{order_id}", headers=_headers()).json(),
            lambda body: body["status"] == "COMPLETED",
            interval_s=1.0,
            timeout_s=20.0,
        )
        assert completed["status"] == "COMPLETED"

        # A completed order can no longer be cancelled.
        cancelled = express_client.delete(f"/orders/{order_id}", headers=_headers())
        assert cancelled.status_code == 409
        assert cancelled.json() == {"error": "Cannot cancel completed order"}


# --------------------------------------------------------------------------
# Order cancellation
# --------------------------------------------------------------------------


class TestOrderCancel:
    def test_cancel_pending_order(self, express_client) -> None:
        created = express_client.post("/orders", json=_order_payload(), headers=_headers()).json()
        response = express_client.delete(f"/orders/{created['orderId']}", headers=_headers())
        assert response.status_code == 200
        body = response.json()
        assert body == {"orderId": created["orderId"], "status": "CANCELLED"}

    def test_cancel_unknown_order(self, express_client) -> None:
        response = express_client.delete("/orders/ORD-99999", headers=_headers())
        assert response.status_code == 404

    def test_cancel_unauthorized(self, express_client) -> None:
        created = express_client.post("/orders", json=_order_payload(), headers=_headers()).json()
        response = express_client.delete(f"/orders/{created['orderId']}", headers={})
        assert response.status_code == 401

    @pytest.mark.slow
    def test_cancel_processing_order_still_allowed(self, express_client) -> None:

        created = express_client.post("/orders", json=_order_payload(), headers=_headers()).json()
        poll_until(
            lambda: express_client.get(f"/orders/{created['orderId']}", headers=_headers()).json(),
            lambda body: body["status"] == "PROCESSING",
            interval_s=0.5,
            timeout_s=10.0,
        )
        response = express_client.delete(f"/orders/{created['orderId']}", headers=_headers())
        assert response.status_code == 200
        assert response.json()["status"] == "CANCELLED"


# --------------------------------------------------------------------------
# Export jobs (60s server-side completion)
# --------------------------------------------------------------------------


class TestExports:
    def test_create_export_job_shape(self, express_client) -> None:
        response = express_client.post("/exports", headers=_headers())
        assert response.status_code == 202
        body = response.json()
        assert JOB_ID_RE.fullmatch(body["jobId"])
        assert body["status"] == "PROCESSING"
        assert body["pollIntervalSeconds"] == 5

    def test_create_export_unauthorized(self, express_client) -> None:
        response = express_client.post("/exports", headers={})
        assert response.status_code == 401

    def test_unknown_export_job(self, express_client) -> None:
        response = express_client.get("/exports/JOB-99999", headers=_headers())
        assert response.status_code == 404
        assert response.json() == {"error": "Export job not found"}

    def test_download_before_completion(self, express_client) -> None:
        job = express_client.post("/exports", headers=_headers()).json()
        response = express_client.get(f"/exports/{job['jobId']}/download", headers=_headers())
        assert response.status_code == 400
        assert response.json() == {"error": "Export file is not ready yet"}

    @pytest.mark.slow
    def test_export_lifecycle_and_csv_download(self, express_client) -> None:

        job = express_client.post("/exports", headers=_headers()).json()
        job_id = job["jobId"]

        premature = express_client.get(f"/exports/{job_id}", headers=_headers()).json()
        assert premature["status"] == "PROCESSING"
        assert premature["downloadUrl"] is None

        completed = poll_until(
            lambda: express_client.get(f"/exports/{job_id}", headers=_headers()).json(),
            lambda body: body["status"] == "COMPLETED",
            interval_s=5.0,
            timeout_s=70.0,
        )
        assert completed["downloadUrl"] == f"/v1/exports/{job_id}/download"

        download = express_client.get(f"/exports/{job_id}/download", headers=_headers())
        assert download.status_code == 200
        assert download.headers["content-type"].startswith("text/csv")
        assert download.headers["content-disposition"] == 'attachment; filename="orders_report.csv"'
        lines = download.text.strip().splitlines()
        assert lines[0] == "orderId,status,totalAmount"
        assert "ORD-10001,COMPLETED,150.00" in lines
        assert "ORD-10002,CANCELLED,45.50" in lines
