"""Focused tests for StorageController session-ledger persistence."""

from __future__ import annotations


class _FakeConfigController:
    def __init__(self, config: dict[str, object]) -> None:
        self._config = config

    def get_config(self) -> dict[str, object]:
        return dict(self._config)


def _build_storage(monkeypatch, tmp_path):
    import config.controller as config_controller
    from storage.controller import StorageController

    var_dir = tmp_path / "var"
    log_dir = tmp_path / "log"
    monkeypatch.setattr(
        config_controller.ConfigController,
        "get_instance",
        lambda: _FakeConfigController({"var_dir": str(var_dir), "log_dir": str(log_dir)}),
    )
    StorageController._instance = None
    return StorageController.get_instance()


def test_session_ledger_tracks_boot_running_shutdown(monkeypatch, tmp_path) -> None:
    from storage.controller import StorageController

    controller = _build_storage(monkeypatch, tmp_path)
    try:
        run_id = controller.get_canonical_run_id()
        controller.mark_session_boot_started(run_id)
        controller.mark_session_running(run_id)
        controller.touch_session_last_seen(run_id)
        controller.mark_session_shutdown_completed(run_id)

        row = controller.session_ledger_conn.execute(
            """
            SELECT lifecycle_state, ready_at, last_seen_at, shutdown_completed_at
            FROM session_ledger
            WHERE canonical_run_id = ?
            """,
            (run_id,),
        ).fetchone()

        assert row is not None
        assert row[0] == "shutdown_completed"
        assert row[1] is not None
        assert row[2] is not None
        assert row[3] is not None
    finally:
        controller.close()
        StorageController._instance = None


def test_previous_run_unclean_detection(monkeypatch, tmp_path) -> None:
    from storage.controller import StorageController

    first = _build_storage(monkeypatch, tmp_path)
    first_run_id = first.get_canonical_run_id()
    first.mark_session_boot_started(first_run_id)
    first.close()
    StorageController._instance = None

    second = _build_storage(monkeypatch, tmp_path)
    try:
        second.mark_session_boot_started(second.get_canonical_run_id())
        prior_unclean, prior = second.previous_run_was_unclean()

        assert prior_unclean is True
        assert prior is not None
        assert prior.canonical_run_id == first_run_id
        assert prior.lifecycle_state == "boot_started"
        assert prior.shutdown_clean is False
        assert prior.shutdown_completed_at is None
    finally:
        second.close()
        StorageController._instance = None


def test_previous_run_clean_detection_uses_shutdown_clean_property(monkeypatch, tmp_path) -> None:
    from storage.controller import StorageController

    first = _build_storage(monkeypatch, tmp_path)
    first_run_id = first.get_canonical_run_id()
    first.mark_session_boot_started(first_run_id)
    first.mark_session_shutdown_completed(first_run_id)
    first.close()
    StorageController._instance = None

    second = _build_storage(monkeypatch, tmp_path)
    try:
        second.mark_session_boot_started(second.get_canonical_run_id())
        prior_unclean, prior = second.previous_run_was_unclean()

        assert prior_unclean is False
        assert prior is not None
        assert prior.shutdown_clean is True
        assert prior.shutdown_completed_at is not None
    finally:
        second.close()
        StorageController._instance = None
