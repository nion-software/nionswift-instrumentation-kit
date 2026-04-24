from __future__ import annotations

import gc
import typing
import unittest

from nion.instrumentation.stem_controller import CameraReservedException
from nion.instrumentation.test import AcquisitionTestContext
from nion.swift.test import TestContext


class _ScopeToken:
    pass


class TestSTEMControllerClass(unittest.TestCase):
    def setUp(self) -> None:
        AcquisitionTestContext.begin_leaks()
        self._test_setup = TestContext.TestSetup()

    def tearDown(self) -> None:
        self._test_setup = typing.cast(typing.Any, None)
        AcquisitionTestContext.end_leaks(self)

    def _test_context(self) -> AcquisitionTestContext.AcquisitionTestContext:
        # subclasses may override this to provide a different configuration
        return AcquisitionTestContext.test_context()

    def test_reserve_and_release_ronchigram_camera(self):
        with self._test_context() as test_context:
            scope_token = _ScopeToken()
            result = test_context.instrument.try_reserve_ronchigram_camera("test", scope_token)
            self.assertTrue(result.is_valid)
            self.assertEqual(1, test_context.instrument._ronchigram_camera_reserved_count)

            test_context.instrument.release_ronchigram_camera()
            self.assertEqual(0, test_context.instrument._ronchigram_camera_reserved_count)

    def test_reserve_ronchigram_camera_twice(self):
        with self._test_context() as test_context:
            scope_token = _ScopeToken()
            result = test_context.instrument.try_reserve_ronchigram_camera("test", scope_token)
            self.assertTrue(result.is_valid)
            self.assertEqual(1, test_context.instrument._ronchigram_camera_reserved_count)

            result2 = test_context.instrument.try_reserve_ronchigram_camera("test2", scope_token)
            self.assertFalse(result2.is_valid)
            self.assertEqual(1, test_context.instrument._ronchigram_camera_reserved_count)

            test_context.instrument.release_ronchigram_camera()

    def test_collected_scope_token_allows_new_reservation(self):
        with self._test_context() as test_context:
            scope_token = _ScopeToken()
            result = test_context.instrument.try_reserve_ronchigram_camera("test", scope_token)
            self.assertTrue(result.is_valid)
            self.assertEqual(1, test_context.instrument._ronchigram_camera_reserved_count)

            test_context.instrument.release_ronchigram_camera()
            self.assertEqual(0, test_context.instrument._ronchigram_camera_reserved_count)

            # Now allow scope_token to be garbage collected and try to reserve again
            # This tests the scenario where the scope token is collected before the second
            # reservation attempt, with _ronchigram_camera_reserved_count already at 0
            scope_token = None
            gc.collect()
            scope_token2 = _ScopeToken()

            result2 = test_context.instrument.try_reserve_ronchigram_camera("test2", scope_token2)
            self.assertTrue(result2.is_valid)
            self.assertEqual(1, test_context.instrument._ronchigram_camera_reserved_count)

            test_context.instrument.release_ronchigram_camera()

    def test_release_without_reservation(self):
        with self._test_context() as test_context:
            # Verify that releasing without a prior reservation is a safe no-op
            self.assertEqual(0, test_context.instrument._ronchigram_camera_reserved_count)
            test_context.instrument.release_ronchigram_camera()
            # Verify state is not corrupted
            self.assertEqual(0, test_context.instrument._ronchigram_camera_reserved_count)

    def test_reservation_failure_contains_correct_exception(self):
        with self._test_context() as test_context:
            scope_token = _ScopeToken()
            result = test_context.instrument.try_reserve_ronchigram_camera("first_task", scope_token)
            self.assertTrue(result.is_valid)

            result2 = test_context.instrument.try_reserve_ronchigram_camera("second_task", scope_token)
            self.assertFalse(result2.is_valid)
            # Verify the exception is CameraReservedException with the correct task_id
            self.assertIsInstance(result2.exception, CameraReservedException)
            self.assertEqual("first_task", result2.exception.task_id)

            test_context.instrument.release_ronchigram_camera()
