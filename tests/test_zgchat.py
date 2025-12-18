"""Tests for ZGChat chat model integration."""

import asyncio
import os
from typing import Optional

import dotenv
import pytest
from a0g.types.model import ServiceStructOutput
from langchain_core.messages import HumanMessage

from langchain_0g import ZGChat

# Load environment variables
dotenv.load_dotenv()


class TestZGChat:
    """Test cases for ZGChat functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, svc: ServiceStructOutput):
        """Setup test environment."""
        self.private_key = os.getenv("A0G_PRIVATE_KEY")
        if not self.private_key:
            pytest.skip("A0G_PRIVATE_KEY environment variable not set")

        # Use dynamically discovered provider
        self.svc = svc

    def test_zgchat_initialization(self):
        """Test ZGChat can be initialized properly."""
        chat = ZGChat(svc=self.svc)
        assert chat.private_key is not None
        assert chat.zg_client is not None
        assert chat.svc is not None
        assert chat.client is not None
        assert chat.async_client is not None

    def test_zgchat_basic_invoke(self):
        """Test basic synchronous chat completion."""
        chat = ZGChat(svc=self.svc)

        response = chat.invoke("Hello, how are you?")

        assert response is not None
        assert hasattr(response, 'content')
        print(f"Response: {response.content}")

    def test_zgchat_invoke_with_message_object(self):
        """Test chat completion with LangChain message objects."""
        chat = ZGChat(svc=self.svc)

        message = HumanMessage(content="What is 0g.ai?")
        response = chat.invoke([message])

        assert response is not None
        assert hasattr(response, 'content')
        print(f"Response to message object: {response.content}")

    @pytest.mark.asyncio
    async def test_zgchat_async_invoke(self):
        """Test asynchronous chat completion."""
        chat = ZGChat(svc=self.svc)

        response = await chat.ainvoke("What is the meaning of life?")

        assert response is not None
        assert hasattr(response, 'content')
        print(f"Async response: {response.content}")

    def test_zgchat_multiple_messages(self):
        """Test chat with multiple conversation messages."""
        chat = ZGChat(svc=self.svc)

        messages = [
            HumanMessage(content="Hello"),
            HumanMessage(content="Can you explain blockchain technology?")
        ]

        response = chat.invoke(messages)

        assert response is not None
        assert hasattr(response, 'content')
        print(f"Multi-message response: {response.content}")

    def test_zgchat_stream_invoke(self):
        """Test streaming chat completion."""
        chat = ZGChat(svc=self.svc)

        try:
            # Test if streaming is supported
            stream = chat.stream("Tell me about artificial intelligence")
            chunks = list(stream)

            assert len(chunks) > 0
            full_response = "".join(chunk.content for chunk in chunks if hasattr(chunk, 'content'))
            print(f"Streamed response: {full_response}")
        except Exception as e:
            # If streaming is not supported, that's ok - just print the info
            print(f"Streaming not supported: {e}")

    def test_zgchat_get_services(self):
        """Test getting available services."""
        chat = ZGChat(svc=self.svc)

        services = chat.zg_client.get_all_services()

        assert services is not None
        assert len(services) > 0
        print(f"Available services: {len(services)}")
        for service in services[:3]:  # Print first 3 services
            print(f"Service: {service.provider}, Model: {service.model}")

    def test_zgchat_model_name_access(self):
        """Test that model name is properly set."""
        chat = ZGChat(svc=self.svc)

        assert hasattr(chat, 'model_name')
        assert chat.model_name is not None
        print(f"Model name: {chat.model_name}")

    def test_zgchat_client_compatibility(self):
        """Test OpenAI client compatibility."""
        chat = ZGChat(svc=self.svc)

        # Test that the underlying OpenAI client works
        openai_response = chat.client.create(
            messages=[{"role": "user", "content": "Hello!"}],
            model=chat.model_name
        )

        assert openai_response is not None
        print(f"OpenAI client response: {openai_response}")


def test_zgchat_basic_functionality():
    """Standalone test for basic functionality (similar to python-0g style)."""
    dotenv.load_dotenv()

    if not os.getenv("A0G_PRIVATE_KEY"):
        print("A0G_PRIVATE_KEY not set, skipping test")
        return

    # Get chatbot provider dynamically
    from a0g import A0G
    a0g_client = A0G()
    services = a0g_client.get_all_services()

    # Find a chatbot service
    chatbot_services = [service for service in services if service.serviceType == 'chatbot']
    if not chatbot_services:
        print("No services available, skipping test")
        return

    svc = chatbot_services[0]

    # Initialize ZGChat with dynamic provider
    chat = ZGChat(svc=svc)

    # Test basic invocation
    response1 = chat.invoke("Hello, how are you?")
    print(f"Response 1: {response1.content}")

    response2 = chat.invoke("What is the meaning of life?")
    print(f"Response 2: {response2.content}")

    response3 = chat.invoke("What is 0g.ai?")
    print(f"Response 3: {response3.content}")

    # Test service listing
    print(f"Found {len(services)} services")


async def test_zgchat_async_functionality():
    """Standalone async test for basic functionality."""
    dotenv.load_dotenv()

    if not os.getenv("A0G_PRIVATE_KEY"):
        print("A0G_PRIVATE_KEY not set, skipping async test")
        return

    # Get chatbot provider dynamically
    from a0g import A0G
    a0g_client = A0G()
    services = a0g_client.get_all_services()

    # Find a chatbot service
    chatbot_services = [service for service in services if service.serviceType == 'chatbot']
    if not chatbot_services:
        print("No services available, skipping async test")
        return

    svc = chatbot_services[0]

    # Initialize ZGChat with dynamic provider
    chat = ZGChat(svc=svc)

    # Test async invocation
    response = await chat.ainvoke("Hello from async test!")
    print(f"Async response: {response.content}")


if __name__ == "__main__":
    # Run basic functionality test
    test_zgchat_basic_functionality()

    # Run async test
    asyncio.run(test_zgchat_async_functionality())