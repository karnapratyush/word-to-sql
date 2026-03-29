"""Entry point for the FastAPI server.

This module creates the FastAPI app and runs it with uvicorn.
It supports both direct execution and uvicorn's module import.

Usage:
    python run_api.py                    # Start on default port 8000
    python run_api.py --port 8080        # Custom port
    python run_api.py --reload           # Dev mode with hot reload
    uvicorn run_api:app --reload         # Direct uvicorn (uses `app` below)
"""

import argparse

import uvicorn

from src.api.main import create_app

# Create the app at module level so uvicorn can import it as "run_api:app"
app = create_app()

if __name__ == "__main__":
    # Parse command-line arguments for server configuration
    parser = argparse.ArgumentParser(description="  AI Logistics API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # Run the server — uses string reference "run_api:app" so reload works
    uvicorn.run(
        "run_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
