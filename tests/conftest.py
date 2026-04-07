"""Shared pytest configuration for ragpipe tests."""


def pytest_addoption(parser):
    parser.addoption(
        "--ragpipe-url",
        default="http://localhost:8090",
        help="ragpipe base URL for live integration tests",
    )
