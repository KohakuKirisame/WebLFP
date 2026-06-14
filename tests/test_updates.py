from weblfp import updates


LOCAL = "1" * 40
REMOTE = "2" * 40


def test_update_check_reports_matching_commits(monkeypatch) -> None:
    monkeypatch.setattr(
        updates,
        "_git_output",
        lambda arguments, timeout=15: LOCAL if arguments[-1] == "HEAD" else f"{LOCAL}\trefs/heads/main",
    )

    result = updates.check_for_updates()

    assert result.status == "up_to_date"
    assert result.update_available is False


def test_update_check_reports_remote_commit_ahead(monkeypatch) -> None:
    monkeypatch.setattr(
        updates,
        "_git_output",
        lambda arguments, timeout=15: LOCAL if arguments[-1] == "HEAD" else f"{REMOTE}\trefs/heads/main",
    )
    monkeypatch.setattr(updates, "_has_commit", lambda commit: commit == REMOTE)
    monkeypatch.setattr(
        updates,
        "_is_ancestor",
        lambda ancestor, descendant: ancestor == LOCAL and descendant == REMOTE,
    )

    result = updates.check_for_updates()

    assert result.status == "update_available"
    assert result.update_available is True
    assert result.latest_commit_url == f"{updates.REPOSITORY_URL}/commit/{REMOTE}"


def test_update_check_does_not_flag_local_commits_as_remote_update(monkeypatch) -> None:
    monkeypatch.setattr(
        updates,
        "_git_output",
        lambda arguments, timeout=15: LOCAL if arguments[-1] == "HEAD" else f"{REMOTE}\trefs/heads/main",
    )
    monkeypatch.setattr(updates, "_has_commit", lambda commit: commit == REMOTE)
    monkeypatch.setattr(
        updates,
        "_is_ancestor",
        lambda ancestor, descendant: ancestor == REMOTE and descendant == LOCAL,
    )

    result = updates.check_for_updates()

    assert result.status == "local_ahead"
    assert result.update_available is False


def test_update_check_handles_unavailable_git_or_network(monkeypatch) -> None:
    monkeypatch.setattr(updates, "_git_output", lambda arguments, timeout=15: None)

    result = updates.check_for_updates()

    assert result.status == "unavailable"
    assert result.update_available is None
