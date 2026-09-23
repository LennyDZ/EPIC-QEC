from epic.core.language import AllocCode, FreeCode


def test_compile_writes_optional_markdown_report_for_gadget_list(
    tmp_path,
    qec_compiler,
    alloc_code_gadget: AllocCode,
    free_code_gadget: FreeCode,
) -> None:
    log_path = tmp_path / "compilation.md"

    qec_compiler.compile(
        [alloc_code_gadget, free_code_gadget], log_output_path=log_path
    )

    report = log_path.read_text(encoding="utf-8")

    assert "# Compilation Report" in report
    assert "Source: list of `QECGadget` objects" in report
    assert "Parsing: succeeded" in report
    assert "![Input QuantumProgram](compilation_input_program.png)" in report
    assert "### Gadget: alloc" in report
    assert "### Gadget: free" in report
    assert "## Output Summary" in report
    assert "- Instructions: 0" in report
    assert "- Detectors: 0" in report
    assert "- Observables: 0" in report
    assert "- Observable tags:\n  - none" in report
    assert (tmp_path / "compilation_input_program.png").is_file()