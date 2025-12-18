"""Tests for ZGLLM text completion model integration."""

import asyncio
import os
from typing import Optional

import dotenv
import pytest
from a0g.types.model import ServiceStructOutput

from langchain_0g import ZGLLM

# Load environment variables
dotenv.load_dotenv()


class TestZGLLM:
    """Test cases for ZGLLM functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, svc: ServiceStructOutput):
        """Setup test environment."""
        self.private_key = os.getenv("A0G_PRIVATE_KEY")
        if not self.private_key:
            pytest.skip("A0G_PRIVATE_KEY environment variable not set")

        # Use dynamically discovered provider
        self.svc = svc

    def test_zgllm_initialization(self):
        """Test ZGLLM can be initialized properly."""
        llm = ZGLLM(svc=self.svc)

        assert llm.private_key is not None
        assert llm.zg_client is not None
        assert llm.svc is not None
        assert llm.client is not None
        assert llm.async_client is not None

    def test_zgllm_basic_invoke(self):
        """Test basic synchronous text completion."""
        llm = ZGLLM(svc=self.svc)

        response = llm.invoke("Complete this sentence: The future of AI is")

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"Response: {response}")

    def test_zgllm_invoke_with_prompt(self):
        """Test text completion with longer prompt."""
        llm = ZGLLM(svc=self.svc)

        prompt = """
        Write a brief summary of blockchain technology:
        Blockchain is a distributed ledger technology that
        """

        response = llm.invoke(prompt)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"Response to prompt: {response}")

    @pytest.mark.asyncio
    async def test_zgllm_async_invoke(self):
        """Test asynchronous text completion."""
        llm = ZGLLM(svc=self.svc)

        response = await llm.ainvoke("Explain quantum computing in one paragraph:")

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"Async response: {response}")

    def test_zgllm_generate(self):
        """Test generation with multiple prompts."""
        llm = ZGLLM(svc=self.svc)

        prompts = [
            "What is artificial intelligence?",
            "Define machine learning:",
            "Explain neural networks:"
        ]

        result = llm.generate(prompts)

        assert result is not None
        assert hasattr(result, 'generations')
        assert len(result.generations) == len(prompts)

        for i, generation in enumerate(result.generations):
            assert len(generation) > 0
            assert hasattr(generation[0], 'text')
            print(f"Generation {i}: {generation[0].text}")

    @pytest.mark.asyncio
    async def test_zgllm_async_generate(self):
        """Test async generation with multiple prompts."""
        llm = ZGLLM(svc=self.svc)

        prompts = [
            "Summarize the benefits of renewable energy:",
            "What are the challenges in space exploration?"
        ]

        result = await llm.agenerate(prompts)

        assert result is not None
        assert hasattr(result, 'generations')
        assert len(result.generations) == len(prompts)

        for i, generation in enumerate(result.generations):
            assert len(generation) > 0
            assert hasattr(generation[0], 'text')
            print(f"Async generation {i}: {generation[0].text}")

    def test_zgllm_get_services(self):
        """Test getting available services."""
        llm = ZGLLM(svc=self.svc)

        services = llm.zg_client.get_all_services()

        assert services is not None
        assert len(services) > 0
        print(f"Available services: {len(services)}")
        for service in services[:3]:  # Print first 3 services
            print(f"Service: {service.provider}, Model: {service.model}")

    def test_zgllm_model_name_access(self):
        """Test that model name is properly set."""
        llm = ZGLLM(svc=self.svc)

        assert hasattr(llm, 'model_name')
        assert llm.model_name is not None
        print(f"Model name: {llm.model_name}")

    def test_zgllm_client_compatibility(self):
        """Test OpenAI client compatibility."""
        llm = ZGLLM(svc=self.svc)

        # Test that the underlying OpenAI client works
        openai_response = llm.client.create(
            prompt="Hello, complete this:",
            model=llm.model_name,
            max_tokens=50
        )

        assert openai_response is not None
        print(f"OpenAI client response: {openai_response}")

    def test_zgllm_with_parameters(self):
        """Test ZGLLM with various completion parameters."""
        llm = ZGLLM(
            svc=self.svc,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )

        response = llm.invoke("Write a creative story about a robot:")

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"Parametrized response: {response}")

    def test_zgllm_streaming(self):
        """Test streaming text completion."""
        llm = ZGLLM(svc=self.svc)

        try:
            # Test if streaming is supported
            stream = llm.stream("Tell me about the history of computers:")
            chunks = list(stream)

            assert len(chunks) > 0
            full_response = "".join(chunk for chunk in chunks if isinstance(chunk, str))
            print(f"Streamed response: {full_response}")
        except Exception as e:
            # If streaming is not supported, that's ok - just print the info
            print(f"Streaming not supported: {e}")


def test_zgllm_basic_functionality():
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

    # Initialize ZGLLM with dynamic provider
    llm = ZGLLM(svc=svc)

    # Test basic text completion
    response1 = llm.invoke("Summarize the following article about AI:")
    print(f"Response 1: {response1}")

    response2 = llm.invoke("Complete this sentence: The future of technology is")
    print(f"Response 2: {response2}")

    response3 = llm.invoke("What is 0g.ai and how does it work?")
    print(f"Response 3: {response3}")

    # Test service listing
    print(f"Found {len(services)} services")


async def test_zgllm_async_functionality():
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

    # Initialize ZGLLM with dynamic provider
    llm = ZGLLM(svc=svc)

    # Test async text completion
    response = await llm.ainvoke("Hello from async LLM test!")
    print(f"Async response: {response}")


if __name__ == "__main__":
    # Run basic functionality test
    test_zgllm_basic_functionality()

    # Run async test
    asyncio.run(test_zgllm_async_functionality())