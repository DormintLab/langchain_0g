# Tests for langchain-0g

This directory contains tests for the langchain-0g package, following the structure and style of the parent python-0g project.

## Setup

1. **Environment Variables**: Copy `.env.example` to `.env` and set your private key:
   ```bash
   cp .env.example .env
   # Edit .env and set A0G_PRIVATE_KEY=your_actual_private_key
   ```

2. **Install Dependencies**: Tests require pytest and other testing dependencies:
   ```bash
   poetry install --with dev
   ```

## Running Tests

### All Tests
```bash
# Using task (recommended)
task tests

# Using pytest directly
pytest

# With verbose output
pytest -v
```

### Specific Test Categories
```bash
# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Run only async tests
pytest -m async_test
```

### Individual Test Files
```bash
# Test ZGChat functionality
pytest tests/test_zgchat.py

# Test ZGLLM functionality
pytest tests/test_zgllm.py

# Run integration tests
pytest tests/test_integration.py
```

### Running Tests in Style of python-0g
```bash
# Run integration test as a script (similar to python-0g style)
python tests/test_integration.py

# Run individual test files as scripts
python tests/test_zgchat.py
python tests/test_zgllm.py
```

## Test Structure

### Test Files

- **`test_zgchat.py`**: Comprehensive tests for ZGChat (chat completion models)
- **`test_zgllm.py`**: Comprehensive tests for ZGLLM (text completion models)
- **`test_integration.py`**: Integration tests demonstrating full functionality
- **`conftest.py`**: Shared pytest fixtures and configuration

### Test Types

1. **Unit Tests**: Test individual components and methods
2. **Integration Tests**: Test real API interactions with 0G.ai services
3. **Async Tests**: Test asynchronous functionality
4. **Error Handling Tests**: Test error scenarios and edge cases

### Test Markers

- `integration`: Tests that use real API calls (all tests in this suite)
- `slow`: Tests that may take longer to run (generation, streaming)
- `async_test`: Async tests using ainvoke, agenerate, etc.

## Test Configuration

### `pytest.ini`
- Configures pytest behavior
- Sets test discovery patterns
- Defines custom markers
- Enables async test support

### `conftest.py`
- Provides shared fixtures for all tests
- Handles environment variable loading
- Automatically skips tests if `A0G_PRIVATE_KEY` is not set
- **Dynamically discovers chatbot providers** from `a0g.get_all_services()` with `serviceType == 'chatbot'`
- Creates test instances of ZGChat and ZGLLM with automatically selected providers
- Falls back to first available service if no chatbot type services are found

### `.env.example`
- Template for environment configuration
- Shows required and optional variables
- Documents expected format for private keys

## Test Examples

### Basic Usage Test
```python
def test_basic_chat(chatbot_provider):
    chat = ZGChat(provider=chatbot_provider)
    response = chat.invoke("Hello!")
    assert response.content
```

### Async Test
```python
@pytest.mark.asyncio
async def test_async_chat():
    chat = ZGChat(provider="0xf07240Efa67755B5311bc75784a061eDB47165Dd")
    response = await chat.ainvoke("Hello!")
    assert response.content
```

### Using Fixtures
```python
def test_with_fixture(zgchat_instance):
    response = zgchat_instance.invoke("Test message")
    assert response.content
```

## Dynamic Provider Discovery

Tests automatically discover available 0G.ai providers at runtime:

1. **Primary**: Look for services with `serviceType == 'chatbot'`
2. **Fallback**: If no chatbot services found, use any available service
3. **Error**: If no services available, tests are skipped

This approach makes tests more resilient and eliminates the need for hardcoded provider addresses.

### Manual Provider Override
If you want to test with a specific provider, you can override the discovery:

```bash
# Set custom provider in environment
export TEST_PROVIDER=0xYourProviderAddressHere
pytest
```

## Notes

- Tests require a valid `A0G_PRIVATE_KEY` environment variable
- All tests are integration tests that make real API calls to 0G.ai services
- **Tests automatically select appropriate providers** from available services
- Tests may be slow depending on network conditions and service response times
- Some tests demonstrate functionality rather than asserting specific outcomes
- Error handling tests intentionally trigger failures to verify proper error handling

## Troubleshooting

### Common Issues

1. **"A0G_PRIVATE_KEY not set"**: Ensure your `.env` file contains a valid private key
2. **Service unavailable errors**: Check if the test provider is online and accessible
3. **Timeout errors**: Some tests may take longer depending on network conditions
4. **Import errors**: Ensure langchain-0g is properly installed (`poetry install`)

### Debug Mode
```bash
# Run with more verbose output
pytest -v -s

# Run with Python debug output
PYTHONPATH=. python -m pytest tests/ -v -s
```