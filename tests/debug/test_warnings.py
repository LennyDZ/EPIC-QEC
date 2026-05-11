from debug.warnings import (
    CompilerWarning,
    MemOverwriteWarning,
    MissingNodeStatusWarning,
    WeirdAllocationWarning,
)


def test_warning_hierarchy() -> None:
    assert issubclass(CompilerWarning, Warning)
    assert issubclass(WeirdAllocationWarning, CompilerWarning)
    assert issubclass(MissingNodeStatusWarning, CompilerWarning)
    assert issubclass(MemOverwriteWarning, CompilerWarning)
