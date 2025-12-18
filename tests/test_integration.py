"""Integration tests for langchain-0g package functionality."""

import asyncio
import os

import dotenv
from langchain_core.messages import HumanMessage

from langchain_0g import ZGChat, ZGLLM

# Load environment variables
dotenv.load_dotenv()


def test_langchain_0g_basic_functionality():
    """Basic integration test demonstrating langchain-0g functionality.

    This test is similar to the demonstration style used in python-0g tests.
    """
    if not os.getenv("A0G_PRIVATE_KEY"):
        print("A0G_PRIVATE_KEY not set, skipping integration test")
        return

    # Get chatbot provider dynamically
    from a0g import A0G
    a0g_client = A0G()
    services = a0g_client.get_all_services()

    # Find a chatbot service
    chatbot_services = [service for service in services if service.serviceType == 'chatbot']
    if not chatbot_services:
        print("No services available, skipping integration test")
        return

    provider = chatbot_services[0].provider
    print(f"Using provider: {provider} (model: {getattr(chatbot_services[0], 'model', 'unknown')})")

    print("=== Testing ZGChat ===")

    # Initialize ZGChat with dynamic provider
    chat = ZGChat(provider=provider)

    # Test chat completions
    response1 = chat.invoke("Hello, how are you?")
    print(f"Chat Response 1: {response1.content}")

    response2 = chat.invoke("What is the meaning of life?")
    print(f"Chat Response 2: {response2.content}")

    response3 = chat.invoke("What is 0g.ai?")
    print(f"Chat Response 3: {response3.content}")

    # Test with LangChain message objects
    message = HumanMessage(content="Explain blockchain in simple terms")
    response4 = chat.invoke([message])
    print(f"Chat Response with Message object: {response4.content}")

    print("\n=== Testing ZGLLM ===")

    # Initialize ZGLLM with dynamic provider
    llm = ZGLLM(provider=provider)

    # Test text completions
    response5 = llm.invoke("Summarize the following article about AI:")
    print(f"LLM Response 1: {response5}")

    response6 = llm.invoke("Complete this sentence: The future of technology is")
    print(f"LLM Response 2: {response6}")

    response7 = llm.invoke("What is 0g.ai and how does it work?")
    print(f"LLM Response 3: {response7}")

    print("\n=== Testing Service Discovery ===")

    # Test getting all available services
    services = chat.zg_client.get_all_services()
    print(f"Found {len(services)} services:")

    for i, service in enumerate(services[:3]):  # Show first 3 services
        print(f"Service {i+1}: Provider={service.provider}, Model={service.model}")

    print("\n=== Testing OpenAI Client Compatibility ===")

    # Test direct OpenAI client usage
    openai_client = chat.zg_client.get_openai_client(chat.provider)
    openai_response = openai_client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello from OpenAI client!"}],
        model=chat.svc.model
    )
    print(f"OpenAI Client Response: {openai_response}")


async def test_langchain_0g_async_functionality():
    """Async integration test for langchain-0g functionality."""
    if not os.getenv("A0G_PRIVATE_KEY"):
        print("A0G_PRIVATE_KEY not set, skipping async integration test")
        return

    print("\n=== Testing Async Functionality ===")

    # Get chatbot provider dynamically
    from a0g import A0G
    a0g_client = A0G()
    services = a0g_client.get_all_services()

    # Find a chatbot service
    chatbot_services = [service for service in services if service.serviceType == 'chatbot']
    if not chatbot_services:
        print("No services available, skipping async integration test")
        return

    provider = chatbot_services[0].provider
    print(f"Using provider for async test: {provider}")

    # Initialize clients with dynamic provider
    chat = ZGChat(provider=provider)
    llm = ZGLLM(provider=provider)

    # Test async chat completion
    async_chat_response = await chat.ainvoke("Hello from async chat!")
    print(f"Async Chat Response: {async_chat_response.content}")

    # Test async text completion
    async_llm_response = await llm.ainvoke("Complete this async prompt:")
    print(f"Async LLM Response: {async_llm_response}")

    # Test async OpenAI client
    async_openai_client = chat.zg_client.get_openai_async_client(chat.provider)
    async_openai_response = await async_openai_client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello from async OpenAI client!"}],
        model=chat.svc.model
    )
    print(f"Async OpenAI Client Response: {async_openai_response}")


def test_langchain_0g_error_handling():
    """Test error handling scenarios."""
    if not os.getenv("A0G_PRIVATE_KEY"):
        print("A0G_PRIVATE_KEY not set, skipping error handling test")
        return

    print("\n=== Testing Error Handling ===")

    try:
        # Test with invalid provider
        invalid_chat = ZGChat(provider="0x0000000000000000000000000000000000000000")
        response = invalid_chat.invoke("This should fail")
        print(f"Unexpected success: {response}")
    except Exception as e:
        print(f"Expected error with invalid provider: {e}")

    try:
        # Test initialization without private key
        os.environ["A0G_PRIVATE_KEY"] = ""
        no_key_chat = ZGChat()
        print(f"Unexpected success without private key")
    except Exception as e:
        print(f"Expected error without private key: {e}")
    finally:
        # Restore private key
        dotenv.load_dotenv()


if __name__ == "__main__":
    # Run all tests when script is executed directly
    print("Running langchain-0g integration tests...")

    # Basic functionality test
    test_langchain_0g_basic_functionality()

    # Async functionality test
    asyncio.run(test_langchain_0g_async_functionality())

    # Error handling test
    test_langchain_0g_error_handling()

    print("\nIntegration tests completed!")