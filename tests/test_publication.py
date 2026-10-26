from tabular_lab.benchmark import curate_run
from tabular_lab.io import append_event, write_json


def test_public_example_excludes_paths_errors_and_prompt_content(tmp_path):
    source, output = tmp_path / "private", tmp_path / "public"
    source.mkdir()
    write_json(source / "summary.json", {"path": "C:/private-data", "status": "partial",
                                         "finalization": {"error": "private-error"}})
    write_json(source / "experiments.json", [{"trial": 0, "spec": {"hypothesis": "safe"},
                                               "status": "failed", "error": "private-error"}])
    write_json(source / "manifest.json", {"seed": 42})
    (source / "report.md").write_text("Safe summary")
    append_event(source / "events.jsonl", {"event": "llm_request", "messages": "private-prompt"})
    append_event(source / "events.jsonl", {"event": "llm_response", "content": "private-response",
                                           "usage": {"prompt_tokens": 12}})
    curate_run(source, output)
    text = "\n".join(p.read_text(encoding="utf-8") for p in output.iterdir())
    assert "private-" not in text
    assert "prompt_tokens" in text
