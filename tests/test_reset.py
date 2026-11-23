import importlib
import threading
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import openarm_mujoco.v2 as assets
import pyarrow as pa
import pytest

main = importlib.import_module("dora_openarm_mujoco.main")


def test_reset_restores_entire_home_keyframe_and_arm_controls():
    model = mujoco.MjModel.from_xml_path(assets.openarm_demo_xml())
    data = mujoco.MjData(model)
    mapper = assets.JointResolver(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert key >= 0
    for _ in range(2):
        data.qpos[:] += 0.1
        data.qvel[:] = 2
        data.ctrl[:] = 0
        data.time = 3
        main._reset_simulation(model, data, mapper, key, True)
        np.testing.assert_array_equal(data.qpos, model.key_qpos[key])
        np.testing.assert_array_equal(data.qvel, model.key_qvel[key])
        assert data.time == model.key_time[key]
        for side in ("right", "left"):
            joints, _ = mapper.get_driver(data.qpos, side)
            assert joints[3] == pytest.approx(np.pi / 2, abs=1e-6)
        expected_ctrl = data.ctrl.copy()
        mapper.set_ctrl(expected_ctrl, main._get_arm_qpos(model, data, "right"), "right")
        mapper.set_ctrl(expected_ctrl, main._get_arm_qpos(model, data, "left"), "left")
        np.testing.assert_array_equal(data.ctrl, expected_ctrl)


def test_reset_requires_a_valid_keyframe():
    with pytest.raises(ValueError, match="keyframe"):
        main._reset_simulation(None, None, None, -1, True)


def test_reset_discards_old_and_future_commands_until_matching_reset():
    node = MagicMock()

    def command(reset_id):
        return {
            "type": "INPUT",
            "id": "move_position_left",
            "value": pa.array(np.zeros(8, dtype=np.float32)),
            "metadata": {"reset_id": reset_id},
        }

    node.__iter__.return_value = [
        command(1),
        {
            "type": "INPUT",
            "id": "reset",
            "value": pa.array(["reset"]),
            "metadata": {"reset_id": 1},
        },
        command(0),
        command(1),
    ]
    with (
        patch.object(main, "_reset_simulation") as reset,
        patch.object(main, "_handle_arm") as move,
    ):
        main._run_dora(
            node, None, None, None, None, threading.Lock(),
            threading.Event(), 0, [], use_ctrl=True,
        )
    reset.assert_called_once()
    move.assert_called_once()
    assert node.send_output.call_args.args[0] == "reset_ready"
    assert node.send_output.call_args.args[1].to_pylist() == [1]
