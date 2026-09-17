"""Shared fixtures for the DriftGate framework's own test suite.

Service-level fixtures (the stateful order & export API, the GenAI demo
service, the long-running poll helper) live with the example suites in
``examples/tests/conftest.py``. Framework unit tests here need no running
service: they exercise the framework directly.
"""
