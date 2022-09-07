import logging
import time
import unittest

from nion.swift import Application
from nion.swift.test import TestContext
from nion.instrumentation.test import AcquisitionTestContext
from nion.ui import TestUI

from nionswift_plugin.nion_instrumentation_ui import MultipleShiftEELSAcquire


class TestMultiAcquire(unittest.TestCase):

    def setUp(self):
        TestContext.begin_leaks()
        self.app = Application.Application(TestUI.UserInterface(), set_global=False)

    def tearDown(self):
        TestContext.end_leaks(self)

    def __test_context(self, *, is_eels: bool = False) -> AcquisitionTestContext.AcquisitionTestContext:
        return AcquisitionTestContext.test_context(is_eels=is_eels)

    def _set_up_multiple_shift_eels_acquire(self):
        return MultipleShiftEELSAcquire.AcquireController()

    def test_multiple_shift_eels_acquire_with_no_offset(self):
        with self.__test_context(is_eels=True) as test_context:
            is_finished = False

            def finished() -> None:
                nonlocal is_finished
                is_finished = True

            logger = logging.getLogger("camera_control_ui")
            logger.propagate = False  # do not send messages to root logger

            stem_controller = test_context.instrument
            eels_camera = test_context.camera_hardware_source
            number_frames = 4
            energy_offset = 0.0
            sleep_time = 0.1
            do_dark_ref = False
            do_cross_correlation = False
            MultipleShiftEELSAcquire.AcquireController().start_threaded_acquire_and_sum(
                stem_controller,
                eels_camera,
                number_frames,
                energy_offset,
                sleep_time,
                do_dark_ref, None,
                do_cross_correlation,
                test_context.document_controller,
                finished
            )

            while not is_finished:
                test_context.document_controller.periodic()
                time.sleep(0.01)


if __name__ == '__main__':
    unittest.main()
