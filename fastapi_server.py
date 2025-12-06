"""
FastAPI Server Entry Point (Legacy)

This file is maintained for backward compatibility.
The application has been refactored into the app/ directory structure.

For new code, use:
- `python app/main.py` or `uvicorn app:app` to run the server
- Import from `app` module instead of `fastapi_server`

This file re-exports the app instance for backward compatibility.
"""

# Import the app from the new structure
from app import app

# Re-export for backward compatibility
__all__ = ['app']
