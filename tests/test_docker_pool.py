"""Tests for the persistent Docker evaluation backend (DockerPool)."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from alife_core.evaluator.docker_pool import DockerPool
from alife_core.models import RunConfig

_OK_RESULT = MagicMock(returncode=0, stdout=b"", stderr=b"")


def _create_result(container_id: str = "abc123") -> MagicMock:
    return MagicMock(returncode=0, stdout=f"{container_id}\n".encode(), stderr=b"")


@pytest.fixture
def default_config() -> RunConfig:
    return RunConfig(sandbox_backend="docker", docker_image="python:3.12-slim")


def _make_pool_with_mock(mock_sub: MagicMock, config: RunConfig, pool_size: int = 1) -> DockerPool:
    """Helper that sets a default side_effect covering create+start per container."""
    if mock_sub.run.side_effect is None:
        effects = []
        for i in range(pool_size):
            effects.append(_create_result(f"container{i}"))  # docker create
            effects.append(_OK_RESULT)  # docker start
        mock_sub.run.side_effect = effects
    pool = DockerPool(config=config, pool_size=pool_size)
    pool.start()
    return pool


class TestDockerPoolLifecycle:
    def test_pool_creates_container_on_start(self, default_config: RunConfig) -> None:
        with patch("alife_core.evaluator.docker_pool.subprocess") as mock_sub:
            pool = _make_pool_with_mock(mock_sub, default_config, pool_size=1)
            try:
                assert len(pool._containers) == 1
                create_call = mock_sub.run.call_args_list[0]
                cmd = create_call[0][0]
                assert "docker" in cmd
                assert "create" in cmd
            finally:
                mock_sub.run.side_effect = None
                mock_sub.run.return_value = _OK_RESULT
                pool.stop()

    def test_pool_applies_security_constraints(self, default_config: RunConfig) -> None:
        with patch("alife_core.evaluator.docker_pool.subprocess") as mock_sub:
            pool = _make_pool_with_mock(mock_sub, default_config, pool_size=1)
            try:
                create_call = mock_sub.run.call_args_list[0]
                cmd = create_call[0][0]
                cmd_str = " ".join(cmd)
                assert "--network" in cmd_str and "none" in cmd_str
                assert "--read-only" in cmd_str
                assert "--cap-drop" in cmd_str and "ALL" in cmd_str
                assert "--security-opt" in cmd_str and "no-new-privileges" in cmd_str
            finally:
                mock_sub.run.side_effect = None
                mock_sub.run.return_value = _OK_RESULT
                pool.stop()

    def test_pool_stop_removes_containers(self, default_config: RunConfig) -> None:
        with patch("alife_core.evaluator.docker_pool.subprocess") as mock_sub:
            pool = _make_pool_with_mock(mock_sub, default_config, pool_size=2)
            mock_sub.run.side_effect = None
            mock_sub.run.return_value = _OK_RESULT
            mock_sub.run.call_args_list.clear()
            pool.stop()
            rm_calls = [call for call in mock_sub.run.call_args_list if "rm" in call[0][0]]
            assert len(rm_calls) == 2

    def test_pool_is_context_manager(self, default_config: RunConfig) -> None:
        with patch("alife_core.evaluator.docker_pool.subprocess") as mock_sub:
            effects = [
                _create_result("ctx0"),
                _OK_RESULT,  # create + start
            ]
            mock_sub.run.side_effect = effects
            with DockerPool(config=default_config, pool_size=1) as pool:
                assert len(pool._containers) == 1
                # Reset for stop() calls
                mock_sub.run.side_effect = None
                mock_sub.run.return_value = _OK_RESULT
            rm_calls = [call for call in mock_sub.run.call_args_list if "rm" in call[0][0]]
            assert len(rm_calls) >= 1


class TestDockerPoolExecution:
    def test_execute_returns_ok_with_valid_output(self, default_config: RunConfig) -> None:
        import base64
        import pickle

        result_payload = ("ok", [("ok", 2)])
        encoded = base64.b64encode(pickle.dumps(result_payload))

        with patch("alife_core.evaluator.docker_pool.subprocess") as mock_sub:
            mock_sub.run.side_effect = [
                _create_result("abc123"),
                _OK_RESULT,  # create + start
                MagicMock(returncode=0, stdout=encoded, stderr=b""),  # docker exec
            ]
            pool = DockerPool(config=default_config, pool_size=1)
            pool.start()
            try:
                status, outputs, detail = pool.execute(
                    code="def solve(x): return x + 1",
                    function_name="solve",
                    cases=(((1,), 2),),
                )
                assert status == "ok"
                assert outputs == [("ok", 2)]
                assert detail == ""
            finally:
                mock_sub.run.side_effect = None
                mock_sub.run.return_value = _OK_RESULT
                pool.stop()

    def test_execute_handles_timeout(self, default_config: RunConfig) -> None:
        with patch("alife_core.evaluator.docker_pool.subprocess") as mock_sub:
            mock_sub.run.side_effect = [
                _create_result("abc123"),
                _OK_RESULT,  # create + start
                subprocess.TimeoutExpired(cmd="docker exec", timeout=1.0),  # exec timeout
            ]
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            pool = DockerPool(config=default_config, pool_size=1)
            pool.start()
            try:
                status, outputs, _detail = pool.execute(
                    code="while True: pass",
                    function_name="solve",
                    cases=(((1,), 2),),
                )
                assert status == "timeout"
                assert outputs is None
            finally:
                mock_sub.run.side_effect = None
                mock_sub.run.return_value = _OK_RESULT
                pool.stop()


class TestDockerPoolRecovery:
    def test_container_crash_triggers_replacement(self, default_config: RunConfig) -> None:
        import base64
        import pickle

        result_payload = ("ok", [("ok", 2)])
        encoded = base64.b64encode(pickle.dumps(result_payload))

        with patch("alife_core.evaluator.docker_pool.subprocess") as mock_sub:
            mock_sub.run.side_effect = [
                # Initial container creation (create + start)
                _create_result("container1"),
                _OK_RESULT,
                # Exec fails (container crashed)
                MagicMock(returncode=137, stdout=b"", stderr=b"container not running"),
                # Health check says container is dead
                MagicMock(returncode=1, stdout=b"", stderr=b""),
                # Remove dead container
                _OK_RESULT,
                # New container creation (create + start)
                _create_result("container2"),
                _OK_RESULT,
                # Retry exec succeeds
                MagicMock(returncode=0, stdout=encoded, stderr=b""),
            ]
            pool = DockerPool(config=default_config, pool_size=1)
            pool.start()
            try:
                status, outputs, _detail = pool.execute(
                    code="def solve(x): return x + 1",
                    function_name="solve",
                    cases=(((1,), 2),),
                )
                assert status == "ok"
                assert outputs == [("ok", 2)]
            finally:
                mock_sub.run.side_effect = None
                mock_sub.run.return_value = _OK_RESULT
                pool.stop()


class TestDockerPoolIntegrationWithConfig:
    def test_use_persistent_docker_flag_in_run_config(self) -> None:
        config = RunConfig(use_persistent_docker=True)
        assert config.use_persistent_docker is True

    def test_use_persistent_docker_defaults_to_false(self) -> None:
        config = RunConfig()
        assert config.use_persistent_docker is False
