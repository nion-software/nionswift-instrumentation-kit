from __future__ import annotations

import typing
import unittest

from nion.instrumentation.test import AcquisitionTestContext
from nion.swift.test import TestContext

class TestScanControlClass(unittest.TestCase):

    def setUp(self):
        AcquisitionTestContext.begin_leaks()
        self._test_setup = TestContext.TestSetup()

    def tearDown(self) -> None:
        self._test_setup = typing.cast(typing.Any, None)
        AcquisitionTestContext.end_leaks(self)

    def _test_context(self) -> AcquisitionTestContext.AcquisitionTestContext:
        # subclasses may override this to provide a different configuration
        return AcquisitionTestContext.test_context()

    def test_acquire_and_release_ronchigram_camera(self):
        with self._test_context() as test_context:
            result = test_context.instrument.try_acquire_ronchigram_camera("test")
            self.assertTrue(result.is_valid)
            self.assertEqual(1, test_context.instrument._ronchigram_camera_acquired_count)

            test_context.instrument.release_ronchigram_camera()
            self.assertEqual(0, test_context.instrument._ronchigram_camera_acquired_count)

    def test_acquire_ronchigram_camera_twice(self):
        with self._test_context() as test_context:
            result = test_context.instrument.try_acquire_ronchigram_camera("test")
            self.assertTrue(result.is_valid)
            self.assertEqual(1, test_context.instrument._ronchigram_camera_acquired_count)

            result2 = test_context.instrument.try_acquire_ronchigram_camera("test2")
            self.assertFalse(result2.is_valid)
            self.assertEqual(1, test_context.instrument._ronchigram_camera_acquired_count)
