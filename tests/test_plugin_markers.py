from __future__ import annotations

import json
from importlib.metadata import entry_points

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch):
    monkeypatch.setenv("COLUMNS", "400")  # keep skip reasons from being truncated


def _plugin_autoloaded() -> bool:
    return any(e.name == "driftgate" for e in entry_points(group="pytest11"))


def _run(pytester, *args):
    load = [] if _plugin_autoloaded() else ["-p", "driftgate.pytest_plugin"]
    return pytester.runpytest_subprocess(*load, "-p", "no:cacheprovider", *args)


def test_missing_dataset_is_skipped_quietly_with_a_how_to(pytester):
    pytester.makepyfile(
        """
        import pytest

        def test_unrelated():
            assert True

        @pytest.mark.golden_dataset("nope.jsonl")
        def test_golden(prompt, golden_response):
            raise AssertionError("must not run")
        """
    )
    result = _run(pytester, "-rs")
    result.assert_outcomes(passed=1, skipped=1)  # no errors, no failures
    result.stdout.fnmatch_lines(["*Golden dataset not found: nope.jsonl*driftgate dataset generate*"])


def test_strict_mode_errors_the_test_instead(pytester):
    (pytester.path / "driftgate.config.yaml").write_text("datasets:\n  on_problem: fail\n")
    pytester.makepyfile(
        """
        import pytest

        def test_unrelated():
            assert True

        @pytest.mark.golden_dataset("nope.jsonl")
        def test_golden(prompt, golden_response): ...
        """
    )
    result = _run(pytester)
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*Golden dataset not found: nope.jsonl*"])


def test_dataset_found_relative_to_test_file_and_parametrized(pytester):
    sub = pytester.mkdir("suite")
    (sub / "data.jsonl").write_text(
        "\n".join(json.dumps({"prompt": f"p{i}", "response": f"r{i}"}) for i in range(3)) + "\n"
    )
    (sub / "test_x.py").write_text(
        "import pytest\n"
        "@pytest.mark.golden_dataset('data.jsonl')\n"
        "def test_golden(prompt, golden_response):\n"
        "    assert golden_response == 'r' + prompt[1:]\n"
    )
    _run(pytester, "suite").assert_outcomes(passed=3)


def test_empty_and_malformed_datasets_skip_with_line_level_reason(pytester):
    (pytester.path / "empty.jsonl").write_text("")
    (pytester.path / "bad.jsonl").write_text('{"prompt": "only prompt"}\n')
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.golden_dataset("empty.jsonl")
        def test_empty(prompt, golden_response): ...

        @pytest.mark.golden_dataset("bad.jsonl")
        def test_bad(prompt, golden_response): ...
        """
    )
    result = _run(pytester, "-rs")
    result.assert_outcomes(skipped=2)
    result.stdout.fnmatch_lines(["*line 1: missing_key: missing response*driftgate dataset check*"])


def test_marker_without_declared_args_does_not_break_collection(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.golden_dataset("whatever.jsonl")
        @pytest.mark.red_team
        def test_plain():
            assert True
        """
    )
    _run(pytester).assert_outcomes(passed=1)


def test_red_team_categories_limit_parametrization(pytester):
    pytester.makepyfile(
        """
        import pytest
        from driftgate.security import ATTACK_CATEGORIES

        @pytest.mark.red_team("tool_abuse", "exfiltration")
        def test_attacks(attack_prompt):
            assert attack_prompt in ATTACK_CATEGORIES["tool_abuse"] + ATTACK_CATEGORIES["exfiltration"]
        """
    )
    expected = 4 + 3  # tool_abuse + exfiltration attacks
    _run(pytester).assert_outcomes(passed=expected)


def test_unknown_red_team_category_is_a_usage_error(pytester):
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.red_team("nope")
        def test_attacks(attack_prompt): ...
        """
    )
    result = _run(pytester)
    assert result.ret != 0
    assert "unknown attack categories" in result.stdout.str() + result.stderr.str()
