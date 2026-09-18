from __future__ import annotations

import os

from pheasant_lab.lifecycle import RunState


def test_state_save_retries_a_transient_windows_file_lock(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    real_replace = os.replace
    attempts = 0

    def locked_twice(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("simulated sync lock")
        real_replace(source, destination)

    monkeypatch.setattr("pheasant_lab.lifecycle.os.replace", locked_twice)
    monkeypatch.setattr("pheasant_lab.lifecycle.time.sleep", lambda _delay: None)

    state = RunState(state_path, {"stage": "evaluate"})
    state.save()

    assert attempts == 3
    assert RunState.load(state_path).data == {"stage": "evaluate"}
