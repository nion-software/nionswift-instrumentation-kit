from __future__ import annotations

import typing
import unittest

from nion.instrumentation.test import AcquisitionTestContext
from nion.swift.test import TestContext


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
            reservation = test_context.instrument.try_reserve_ronchigram_camera("test")
            self.assertIsNotNone(reservation)
            self.assertIsNotNone(reservation.camera)
            self.assertEqual("test", reservation.task_id)
            self.assertIsNone(reservation.failure_reason)
            self.assertIsNotNone(test_context.instrument._reserved_ronchigram_camera)

            reservation = None
            self.assertIsNone(test_context.instrument._reserved_ronchigram_camera())

    def test_reserve_and_out_of_scope_release_ronchigram_camera(self):
        with self._test_context() as test_context:
            with test_context.instrument.try_reserve_ronchigram_camera("test") as reservation:
                self.assertIsNotNone(reservation)
                self.assertIsNotNone(reservation.camera)
                self.assertEqual("test", reservation.task_id)
                self.assertIsNone(reservation.failure_reason)
                self.assertIsNotNone(test_context.instrument._reserved_ronchigram_camera)

            self.assertIsNone(test_context.instrument._reserved_ronchigram_camera)

    def test_reserve_ronchigram_camera_twice(self):
        with self._test_context() as test_context:
            reservation = test_context.instrument.try_reserve_ronchigram_camera("test")
            self.assertIsNotNone(reservation)
            self.assertIsNotNone(reservation.camera)
            self.assertEqual("test", reservation.task_id)
            self.assertIsNone(reservation.failure_reason)
            self.assertIsNotNone(test_context.instrument._reserved_ronchigram_camera)

            reservation2 = test_context.instrument.try_reserve_ronchigram_camera("test2")
            self.assertIsNone(reservation2.camera)
            self.assertIsNotNone(reservation2.failure_reason)
            self.assertIn("test", reservation2.failure_reason)

            reservation = None
            reservation2 = None

    def test_leaving_scope_allows_new_reservation(self):
        with self._test_context() as test_context:
            with test_context.instrument.try_reserve_ronchigram_camera("test") as reservation:
                self.assertIsNotNone(reservation)
                self.assertIsNotNone(reservation.camera)
                self.assertEqual("test", reservation.task_id)
                self.assertIsNone(reservation.failure_reason)
                self.assertIsNotNone(test_context.instrument._reserved_ronchigram_camera)

            reservation2 = test_context.instrument.try_reserve_ronchigram_camera("test2")
            self.assertIsNotNone(reservation2)
            self.assertIsNotNone(reservation2.camera)
            self.assertIsNone(reservation2.failure_reason)

            reservation2 = None
