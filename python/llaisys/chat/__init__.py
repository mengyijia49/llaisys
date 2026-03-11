def build_app(*args, **kwargs):
    from .server import build_app as _build_app

    return _build_app(*args, **kwargs)


__all__ = ["build_app"]
