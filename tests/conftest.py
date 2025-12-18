"""Pytest configuration and fixtures for langchain-0g tests."""

import os
from typing import Generator, Optional

import dotenv
import pytest
from a0g import A0G


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables for all tests."""
    dotenv.load_dotenv()


@pytest.fixture(scope="session")
def private_key() -> Optional[str]:
    """Provide private key for testing."""
    return os.getenv("A0G_PRIVATE_KEY")


@pytest.fixture(scope="session")
def a0g_client(private_key: Optional[str]) -> A0G:
    """Create A0G client for testing."""
    if not private_key:
        pytest.skip("A0G_PRIVATE_KEY environment variable not set")
    return A0G(private_key=private_key)


@pytest.fixture(scope="session")
def chatbot_provider(a0g_client: A0G) -> str:
    """Get a chatbot provider dynamically from available services."""
    services = a0g_client.get_all_services()

    # Find a service with type "chatbot"
    chatbot_services = [service for service in services if service.serviceType == 'chatbot']

    if not chatbot_services:
        pytest.skip("No chatbot services available")

    provider = chatbot_services[0].provider
    print(f"Using provider: {provider} (model: {getattr(chatbot_services[0], 'model', 'unknown')})")
    return provider


@pytest.fixture(scope="session")
def test_provider(chatbot_provider: str) -> str:
    """Provide test provider address (alias for backward compatibility)."""
    return chatbot_provider


@pytest.fixture(autouse=True)
def skip_if_no_private_key(private_key: Optional[str]):
    """Skip tests if A0G_PRIVATE_KEY is not set."""
    if not private_key:
        pytest.skip("A0G_PRIVATE_KEY environment variable not set")


@pytest.fixture
def zgchat_instance(chatbot_provider: str):
    """Create a ZGChat instance for testing."""
    from langchain_0g import ZGChat
    return ZGChat(provider=chatbot_provider)


@pytest.fixture
def zgllm_instance(chatbot_provider: str):
    """Create a ZGLLM instance for testing."""
    from langchain_0g import ZGLLM
    return ZGLLM(provider=chatbot_provider)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "async_test: mark test as async")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add integration marker to all tests since they use real API
        item.add_marker(pytest.mark.integration)

        # Add async marker to async tests
        if "async" in item.name or "ainvoke" in item.name:
            item.add_marker(pytest.mark.async_test)

        # Add slow marker to generate tests (they typically take longer)
        if "generate" in item.name or "stream" in item.name:
            item.add_marker(pytest.mark.slow)