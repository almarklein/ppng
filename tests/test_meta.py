import ppng


def test_namespace():
    names = set(ppng.ppngwriter.__all__)
    names |= set(ppng.ppngreader.__all__)
    namespace = set(dir(ppng))
    for name in names:
        assert name in namespace
    for name in {"__version__", "version_info"}:
        assert name in namespace
    for name in namespace:
        if not (
            name.startswith("_") or name in {"version_info", "ppngwriter", "ppngreader"}
        ):
            assert name in names
