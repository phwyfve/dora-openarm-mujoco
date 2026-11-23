# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import threading
import unittest
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pyarrow as pa

main = importlib.import_module("dora_openarm_mujoco.main")


class SnapshotTimestampTests(unittest.TestCase):
    def test_synchronous_actions_advance_only_after_both_arms(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep="0.001"/><worldbody>'
            '<body><joint name="openarm_right_joint1"/><geom size="0.1" mass="1"/>'
            '</body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        node = MagicMock()
        mapper = MagicMock()
        target = pa.array(
            [{"qpos": np.zeros(8, dtype=np.float32)}], type=main._QPOS_TYPE
        )

        def events():
            for step in range(30):
                previous_time = data.time
                yield {
                    "type": "INPUT", "id": "move_position_right",
                    "value": target, "metadata": {"control_step": step},
                }
                self.assertEqual(data.time, previous_time)
                self.assertEqual(mapper.set_ctrl.call_count, step * 2)
                yield {"type": "INPUT", "id": "ui_tick"}
                self.assertEqual(data.time, previous_time)
                yield {
                    "type": "INPUT", "id": "move_position_left",
                    "value": target, "metadata": {"control_step": step},
                }
                self.assertLessEqual(abs(data.time - (step + 1) / 30), 0.0005 + 1e-12)

        node.__iter__.return_value = events()
        scheduler = MagicMock()
        main._run_synchronous(
            node, model, data, mapper, threading.Event(), 30,
            threading.Lock, cam_scheduler=scheduler,
        )
        self.assertAlmostEqual(data.time, 1.0, places=10)
        self.assertEqual(mapper.set_ctrl.call_count, 60)
        calls = node.send_output.call_args_list
        self.assertEqual(len(calls), 31 * 3)
        for step in range(31):
            snapshot = calls[step * 3:step * 3 + 3]
            self.assertEqual([call.args[0] for call in snapshot],
                             ["state_right", "state_left", "control_tick"])
            metadata = snapshot[0].kwargs["metadata"]
            self.assertEqual(metadata["control_step"], step)
            self.assertTrue(all(call.kwargs["metadata"] == metadata for call in snapshot))
            self.assertEqual(scheduler.publish.call_args_list[step].args[1], metadata)

    def test_synchronous_mismatched_action_never_advances_physics(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><joint/><geom size="0.1" mass="1"/>'
            '</body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        node = MagicMock()
        mapper = MagicMock()
        target = pa.array([{"qpos": [0.0] * 8}], type=main._QPOS_TYPE)
        node.__iter__.return_value = [
            {"type": "INPUT", "id": f"move_position_{side}",
             "value": target, "metadata": {"control_step": step}}
            for side, step in (("right", 0), ("left", 1))
        ]
        with self.assertRaisesRegex(ValueError, "Expected control_step=0"):
            main._run_synchronous(
                node, model, data, mapper, threading.Event(), 30, threading.Lock,
            )
        self.assertEqual(data.time, 0)
        mapper.set_ctrl.assert_not_called()

    def test_request_outputs_and_snapshot_timestamps(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><joint name="openarm_right_joint1"/>'
            '<geom size="0.1" mass="1"/></body></worldbody></mujoco>'
        )
        state = main._get_arm_state(model, mujoco.MjData(model), "right")
        self.assertEqual(state["motor_status"], ["ENABLED"] + ["SILENT"] * 7)
        bus = {
            "carrier": True,
            "bus_off": 0,
            "error_passive": 0,
            "error_warning": 0,
            "ack_error": 0,
            "tx_overflow": 0,
            "rx_overflow": 0,
            "net_down": 0,
        }
        self.assertEqual(state["bus"], bus)
        for kind, getter, value in (
            ("position", "_get_arm_qpos", state["qpos"]),
            ("state", "_get_arm_state", state),
        ):
            with self.subTest(kind=kind):
                metadata = {"timestamp": 10, "trace_id": "preserve"}
                event = {"type": "INPUT", "id": f"request_{kind}", "metadata": metadata}
                node = MagicMock()
                node.__iter__.return_value = [event, event]
                lock = threading.Lock()
                times = iter((100, 200))

                def snapshot_time():
                    self.assertTrue(lock.locked())
                    self.assertEqual(read.call_count, 2 * clock.call_count)
                    return next(times)

                with (
                    patch.object(main, getter, return_value=value) as read,
                    patch.object(
                        main.time, "time_ns", side_effect=snapshot_time
                    ) as clock,
                ):
                    main._run_dora(
                        node, None, None, None, None, lock, threading.Event(), -1, []
                    )
                calls = node.send_output.call_args_list
                self.assertEqual(
                    [call.args[0] for call in calls],
                    [f"{kind}_right", f"{kind}_left"] * 2,
                )
                for call, timestamp in zip(calls, (100, 100, 200, 200)):
                    self.assertEqual(
                        call.kwargs["metadata"],
                        {"trace_id": "preserve", "observation_timestamp": timestamp},
                    )
                    if kind == "state":
                        output = call.args[1]
                        self.assertEqual(
                            output[0].as_py()["motor_status"], state["motor_status"]
                        )
                        self.assertEqual(output[0].as_py()["bus"], bus)
                        self.assertEqual(
                            output.type.field("motor_status").type,
                            pa.list_(pa.string()),
                        )
                        for field in output.type.field("bus").type:
                            self.assertEqual(
                                field.type,
                                pa.bool_() if field.name == "carrier" else pa.int64(),
                            )
                self.assertEqual(metadata, {"timestamp": 10, "trace_id": "preserve"})
                self.assertEqual(clock.call_count, 2)
