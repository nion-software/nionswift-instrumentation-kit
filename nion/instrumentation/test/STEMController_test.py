from __future__ import annotations

import unittest
import typing

from nion.instrumentation import stem_controller
from nion.instrumentation.test import AcquisitionTestContext
from nion.swift.test import TestContext
from nion.utils import Stream


class TestModeControlStream(unittest.TestCase):
    def setUp(self):
        AcquisitionTestContext.begin_leaks()
        self._test_setup = TestContext.TestSetup()

    def tearDown(self):
        self._test_setup = typing.cast(typing.Any, None)
        AcquisitionTestContext.end_leaks(self)

    def _test_context(self) -> AcquisitionTestContext.AcquisitionTestContext:
        # subclasses may override this to provide a different configuration
        return AcquisitionTestContext.test_context()

    def test_mode_control_stream_combines_mode_and_control(self) -> None:
        with self._test_context() as test_context:
            controller = test_context.instrument

            stream = controller.get_mode_control_stream("track-control", "C10")
            events: list[stem_controller.ModeControlSession] = []
            listener = stream.value_stream.listen(lambda value: events.append(value))
            self.assertIsNone(stream.value)
            self.assertEqual(len(events), 0)

            with controller.active_mode("track-control", {"control": "C10"}):
                controller.SetVal("C10", 1.5)

                self.assertIsNotNone(stream.value)
                self.assertEqual(stream.value.mode_id, "track-control")
                self.assertEqual(stream.value.control_name, "C10")
                self.assertEqual(stream.value.control_try_value.value, 1.5)
                self.assertTrue(stream.value.control_try_value.is_valid)
                self.assertEqual(stream.value.control_value, 1.5)
                self.assertTrue(stream.value.active)

                controller.SetVal("C10", 2.25)
                self.assertEqual(stream.value.control_try_value.value, 2.25)
                self.assertEqual(events[-1].control_try_value.value, 2.25)
                self.assertEqual(stream.value.control_value, 2.25)

            self.assertFalse(stream.value.active)
            self.assertEqual(stream.value.end_reason, stem_controller.ModeEndReason.COMMITTED)
            self.assertEqual(stream.value.control_try_value.value, 2.25)
            self.assertEqual(stream.value.control_value, 2.25)
            self.assertEqual(events[-1].end_reason, stem_controller.ModeEndReason.COMMITTED)

            control_stream = typing.cast(Stream.ValueStream[stem_controller.TryValue[float]], typing.cast(typing.Any, controller.get_control_try_value_stream("C10")))
            control_stream.send_value(stem_controller.TryValue(None, Exception("control unavailable")))
            self.assertEqual(stream.value.control_try_value.value, 2.25)
            self.assertEqual(events[-1].control_try_value.value, 2.25)
            listener.close()

    def test_active_mode_exits_when_body_raises(self) -> None:
        with self._test_context() as test_context:
            controller = test_context.instrument

            stream = controller.get_mode_control_stream("track-control", "C10")
            events: list[stem_controller.ModeControlSession] = []
            listener = stream.value_stream.listen(lambda value: events.append(value))

            with self.assertRaisesRegex(RuntimeError, "boom"):
                with controller.active_mode("track-control", {"control": "C10"}):
                    controller.SetVal("C10", 3.0)
                    self.assertIsNotNone(stream.value)
                    self.assertTrue(stream.value.active)
                    raise RuntimeError("boom")

            self.assertIsNotNone(stream.value)
            self.assertFalse(stream.value.active)
            self.assertEqual(stream.value.end_reason, stem_controller.ModeEndReason.COMMITTED)
            self.assertTrue(any(event.active for event in events))
            self.assertFalse(events[-1].active)
            listener.close()

    def test_mode_entry_can_drive_overlay_lifecycle_for_matching_control(self) -> None:
        class _ModeOverlayController:
            def __init__(self, controller: stem_controller.STEMController, mode_stream_id: str, control_filter: str) -> None:
                self.__controller = controller
                self.__mode_stream_id = mode_stream_id
                self.__control_filter = control_filter
                self.__active_mode_session_id: str | None = None
                self.__mode_control_stream: Stream.AbstractStream[stem_controller.ModeControlSession] | None = None
                self.__mode_control_listener: typing.Any = None
                self.overlay_value: float | None = None
                self.overlay_active = False
                self.actions: list[str] = []
                mode_stream = controller.get_mode_stream(mode_stream_id)
                self.__mode_listener = mode_stream.value_stream.listen(self.__mode_changed)

            def close(self) -> None:
                if self.__mode_control_listener:
                    self.__mode_control_listener.close()
                    self.__mode_control_listener = None
                self.__mode_control_stream = None
                self.__mode_listener.close()

            def __mode_changed(self, mode_session: stem_controller.ModeSession | None) -> None:
                if mode_session is None:
                    return
                if mode_session.active:
                    control_name = typing.cast(str | None, mode_session.payload.get("control"))
                    if mode_session.mode_id == self.__mode_stream_id and control_name == self.__control_filter:
                        self.__active_mode_session_id = mode_session.mode_session_id
                        self.__mode_control_stream = self.__controller.get_mode_control_stream(self.__mode_stream_id, control_name)
                        self.__mode_control_listener = self.__mode_control_stream.value_stream.listen(self.__mode_control_changed)
                        self.overlay_active = True
                        self.actions.append("create")
                    return
                if mode_session.mode_session_id == self.__active_mode_session_id:
                    if self.__mode_control_listener:
                        self.__mode_control_listener.close()
                        self.__mode_control_listener = None
                    self.__mode_control_stream = None
                    self.__active_mode_session_id = None
                    self.overlay_value = None
                    self.overlay_active = False
                    self.actions.append("remove")

            def __mode_control_changed(self, mode_control_session: stem_controller.ModeControlSession | None) -> None:
                if mode_control_session is None:
                    return
                if not mode_control_session.active:
                    return
                if mode_control_session.mode_session_id != self.__active_mode_session_id:
                    return
                control_value = mode_control_session.control_try_value.value
                if mode_control_session.control_try_value.is_valid and control_value is not None:
                    self.overlay_value = control_value
                    self.actions.append(f"update:{control_value}")

        with self._test_context() as test_context:
            controller = test_context.instrument
            overlay_controller = _ModeOverlayController(controller, "track-control", "C10")
            try:
                with controller.active_mode("track-control", {"control": "C12"}):
                    controller.SetVal("C12", 1.0)
                    self.assertFalse(overlay_controller.overlay_active)
                    self.assertEqual(overlay_controller.actions, [])

                with controller.active_mode("track-control", {"control": "C10"}):
                    controller.SetVal("C10", 4.5)
                    self.assertTrue(overlay_controller.overlay_active)
                    self.assertEqual(overlay_controller.overlay_value, 4.5)
                    controller.SetVal("C10", 7.25)
                    self.assertEqual(overlay_controller.overlay_value, 7.25)

                self.assertFalse(overlay_controller.overlay_active)
                self.assertIsNone(overlay_controller.overlay_value)
                self.assertEqual(overlay_controller.actions, ["create", "update:4.5", "update:7.25", "remove"])
            finally:
                overlay_controller.close()


if __name__ == "__main__":
    unittest.main()


