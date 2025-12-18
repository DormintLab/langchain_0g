# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python library that provides LangChain integrations for 0G.ai blockchain-based AI services. The library enables developers to use 0G.ai hosted LLMs through LangChain's familiar interface while handling blockchain authentication via wallet private keys.

## Development Commands

### Setup
```bash
# Install dependencies using Poetry
poetry install

# Set environment variable for wallet authentication
export A0G_PRIVATE_KEY="your_wallet_private_key"
```

### Code Quality and Formatting
```bash
# Format code (using Ruff and isort)
task format

# Lint code and auto-fix issues
task lint

# Type checking with MyPy
task typecheck

# Clean __pycache__ directories
task clean

# Run all checks (format, clean, lint)
task all
```

### Testing
```bash
# Run tests in Docker
task tests
# Note: This executes: sudo docker compose exec backend bash -c 'PYTHONPATH=. pytest -v'
```

### Building and Publishing
```bash
# Build package for distribution
poetry build

# Publishing is automated via GitHub Actions on version tags (v*)
git tag v0.1.1
git push origin v0.1.1
```

## Architecture

### Core Components

The library provides two main LangChain-compatible classes:

1. **ZGChat** (`langchain_0g/chat_models/base.py:15`): Modern chat completion interface that wraps 0G.ai chat models
2. **ZGLLM** (`langchain_0g/llms/`): Legacy text completion interface for older model types

### Authentication Pattern

Unlike traditional API key authentication, this library uses blockchain wallet authentication:
- Requires `A0G_PRIVATE_KEY` environment variable containing a wallet private key
- All requests are signed and verified through 0G.ai smart contracts
- ENS addresses identify model providers (e.g., "0xf07240Efa67755B5311bc75784a061eDB47165Dd")

### Client Architecture

The `ZGChat` class implements a sophisticated client pattern:
- Inherits from LangChain's `ChatOpenAI` for compatibility
- Overrides authentication to use wallet-based signing via `A0G` client
- Provides both sync (`client`) and async (`async_client`) OpenAI-compatible interfaces
- Automatically sets dummy API keys to satisfy Pydantic validation requirements

### Key Integration Points

- **A0G SDK**: Core blockchain client (`from a0g import A0G`)
- **LangChain Core**: Inherits from `ChatOpenAI` for seamless integration
- **Web3 Types**: Uses ENS typing for provider addresses
- **OpenAI Compatibility**: Maintains OpenAI client interface through adapter pattern

## Development Notes

### Package Structure
```
langchain_0g/
├── __init__.py          # Public exports (ZGChat, ZGLLM)
├── chat_models/
│   └── base.py          # ZGChat implementation
└── llms/
    └── base.py          # ZGLLM implementation
```

### Dependencies
- **Core**: `python-0g[langchain]` (0.6.1.1 to 0.7.0.0)
- **Python**: 3.10 to 3.99
- **Build**: Poetry-based with standard wheel distribution

### CI/CD Pipeline
- Automated publishing to PyPI and GitHub Releases
- Triggered by version tags starting with 'v*'
- Uses Poetry for dependency management and building

### Environment Setup
Create a `.env` file with:
```
A0G_PRIVATE_KEY=your_wallet_private_key_here
```

### Testing Strategy
Tests are designed to run in a Docker environment with a backend service. The test command assumes a Docker Compose setup with a `backend` service where tests can be executed.