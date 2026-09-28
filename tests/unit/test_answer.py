"""Answer extraction and normalization (κ)."""

from sage_mas.answer.boxed import ends_with_boxed_answer, ensure_boxed, last_boxed_answer
from sage_mas.answer.extractor import RegexExtractor
from sage_mas.answer.normalize import answer_key
from sage_mas.answer.regex import extract_final_answer, extract_ground_truth_answer
from sage_mas.answer.xfinder import answer_range, parse_output


def test_answer_keys():
    assert answer_key("1,000") == answer_key("1000") == "num:1000"
    assert answer_key("0.50") == answer_key("1/2") == answer_key("50%") == "num:1/2"
    assert answer_key("$5") == "num:5"
    assert answer_key("(B)") == answer_key("B.") == "choice:b"
    assert answer_key("\\frac{3}{4}") == "num:3/4"
    assert answer_key("x^2 + 1") == "text:x^2+1"
    assert answer_key("") == answer_key(None) == ""


def test_boxed_helpers():
    assert last_boxed_answer("a \\boxed{1} b \\boxed{2}") == "2"
    assert last_boxed_answer("\\boxed{...}") == ""
    assert last_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert ends_with_boxed_answer("steps\n\\boxed{4}\n")
    assert not ends_with_boxed_answer("\\boxed{4}\nmore text")


def test_ensure_boxed():
    assert ensure_boxed("x\n\\boxed{3}", lambda t: "9") == "x\n\\boxed{3}"
    assert ensure_boxed("so it is 7 ", lambda t: "7") == "so it is 7\n\nFinal answer: \\boxed{7}"
    assert ensure_boxed("nothing", lambda t: "") == "nothing"
    assert ensure_boxed("   ", lambda t: "1") == ""


def test_regex_extraction():
    assert extract_final_answer("The answer is 42.") == "42."
    assert extract_final_answer("Final answer: \\boxed{17}") == "17"
    assert extract_final_answer("C) the third one") == "C) the third one"
    assert RegexExtractor().extract("q", [], "so the total is 12 apples").key == "num:12"
    assert extract_ground_truth_answer("work\n#### 72") == "72"


def test_xfinder_parsing_and_range():
    assert parse_output(" Key extracted answer: 18<|endoftext|>") == ("18", False)
    assert parse_output("[No valid answer]") == ("", True)
    assert parse_output('"B"') == ("B", False)
    assert answer_range(["x", "y"]) == '[["A", "x"], ["B", "y"]]'
    assert answer_range(None).startswith("a(n) number")
