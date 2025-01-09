import ppng


def test_namespace():
    names = set(ppng.ppng.__all__)
    namespace = set(dir(ppng))
    for name in names:
        assert name in namespace
    for name in {"__version__", "version_info"}:
        assert name in namespace
    for name in namespace:
        if not (name.startswith("_") or name in {"version_info", "ppng"}):
            assert name in names
