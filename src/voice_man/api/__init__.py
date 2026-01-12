"""
Voice Man API Package

Contains all API routers and the main FastAPI application.
"""

# Lazy import to avoid import errors during testing
__all__ = ["app"]


def __getattr__(name):
    if name == "app":
        from voice_man.api.main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
