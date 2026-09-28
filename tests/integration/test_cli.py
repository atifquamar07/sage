"""End to end: sage-run -> sage-eval -> sage-aggregate against a fake OpenAI-compatible server."""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sage_mas import aggregate, evaluate, run
from sage_mas.prompts.templates import CRITIC_SYSTEM, REWRITER_SYSTEM


def _chat_reply(messages: list[dict]) -> str:
    system = messages[0]["content"] if messages[0]["role"] == "system" else None
    user = messages[-1]["content"]
    if system == CRITIC_SYSTEM:
        return "action: EDIT\nfinal_answer:\nChecked again.\nFinal answer: \\boxed{4}"
    if system == REWRITER_SYSTEM:
        current = user.split("CURRENT_SYSTEM_PROMPT_BEGIN\n")[1].split("\nCURRENT_SYSTEM_PROMPT_END")[0]
        return current + "\n\nRe-derive every intermediate value independently before stating the final result."
    if system is None:  # xVerify
        output = re.search(r'Output sentence: """(.*?)"""', user, re.S).group(1)
        reference = re.search(r"Correct answer: (.*)\n", user).group(1)
        return "Correct" if output.strip() == reference.strip() else "Incorrect"
    return "Adding the numbers step by step gives four.\n\\boxed{4}"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/chat/completions"):
            text = _chat_reply(body["messages"])
            choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            payload = {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [choice],
            }
        else:  # xFinder uses the raw completions endpoint
            boxed = re.findall(r"\\boxed\{([^}]*)\}", body["prompt"])
            text = boxed[-1] if boxed else "[No valid answer]"
            payload = {
                "id": "y",
                "object": "text_completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            }
        payload["usage"] = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def test_run_evaluate_aggregate(tmp_path, server):
    for name, extra in [
        ("task", "sampling: {top_p: 0.8}\n"),
        ("xfinder", "max_tokens: 100\n"),
        ("xverify", "max_tokens: 8\n"),
    ]:
        (tmp_path / f"{name}.yaml").write_text(f"name: {name}\nendpoints: [{server}]\n{extra}")
    dataset = tmp_path / "toy.json"
    dataset.write_text(
        json.dumps(
            [
                {"sample_id": f"q{i}", "query": f"What is 2 + 2? ({i})", "gt": "#### 4" if i < 2 else "#### 5"}
                for i in range(3)
            ]
        )
    )
    common = [
        "--model",
        str(tmp_path / "task.yaml"),
        "--xfinder",
        str(tmp_path / "xfinder.yaml"),
        "--dataset",
        str(dataset),
        "--out",
        str(tmp_path / "results"),
        "--workers",
        "3",
    ]

    for seed in (1, 2):
        assert run.main([*common, "--seed", str(seed)]) == 0
    results = tmp_path / "results" / "task" / "seed_1" / "toy.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(rows) == 3
    row = rows[0]
    assert row["answer"].endswith("\\boxed{4}") and row["answer_key"] == "num:4"
    assert {c["stage"] for c in row["calls"]} >= {"initial_generation", "reciprocal_comparison", "prompt_rewriting"}
    assert [r["status"] for r in row["trace"]["rewrites"]].count("accepted") == 3
    assert "gt" not in row

    assert run.main([*common, "--seed", "1"]) == 0  # resume: nothing left to do
    assert len(results.read_text().splitlines()) == 3
    with pytest.raises(SystemExit):  # different settings in the same folder are refused
        run.main([*common, "--seed", "1", "--set", "routing.top_k=1"])

    eval_args = ["--xfinder", str(tmp_path / "xfinder.yaml"), "--judge", str(tmp_path / "xverify.yaml")]
    for seed in (1, 2):
        path = tmp_path / "results" / "task" / f"seed_{seed}" / "toy.jsonl"
        assert evaluate.main([str(path), *eval_args]) == 0
    table = aggregate.collect(tmp_path / "results" / "task")
    assert table == {"toy": {1: pytest.approx(200 / 3), 2: pytest.approx(200 / 3)}}
