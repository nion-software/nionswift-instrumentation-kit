from __future__ import annotations

import asyncio
import gc
import threading
import typing
import unittest

from nion.instrumentation.test import AcquisitionTestContext
from nion.instrumentation import stem_controller
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

    def test_reserve_and_release_ronchigram_camera(self) -> None:
        with self._test_context() as test_context:
            self.assertTrue(test_context.instrument.is_ronchigram_available.value)
            reservation = test_context.instrument._try_reserve_ronchigram_camera()
            self.assertIsNotNone(reservation)
            assert reservation
            self.assertIsNotNone(reservation.resource)
            self.assertFalse(test_context.instrument.is_ronchigram_available.value)

            reservation.release()
            self.assertTrue(test_context.instrument.is_ronchigram_available.value)

    def test_reserve_and_release_ronchigram_camera_via_context_manager(self) -> None:
        with self._test_context() as test_context:
            with test_context.instrument._try_reserve_ronchigram_camera() as reservation:
                self.assertIsNotNone(reservation)
                assert reservation
                self.assertIsNotNone(reservation.resource)
                self.assertFalse(test_context.instrument.is_ronchigram_available.value)

            self.assertTrue(test_context.instrument.is_ronchigram_available.value)

    def test_reserve_ronchigram_camera_twice(self) -> None:
        with self._test_context() as test_context:
            reservation = test_context.instrument._try_reserve_ronchigram_camera()
            self.assertIsNotNone(reservation)
            assert reservation
            self.assertIsNotNone(reservation.resource)

            reservation2 = test_context.instrument._try_reserve_ronchigram_camera()
            self.assertIsNone(reservation2)

            reservation.release()

    def test_leaving_with_scope_allows_new_reservation(self) -> None:
        with self._test_context() as test_context:
            with test_context.instrument._try_reserve_ronchigram_camera() as reservation:
                self.assertIsNotNone(reservation)
                assert reservation
                self.assertIsNotNone(reservation.resource)

            reservation2 = test_context.instrument._try_reserve_ronchigram_camera()
            self.assertIsNotNone(reservation2)
            assert reservation2
            self.assertIsNotNone(reservation2.resource)

            reservation2.release()

    def test_unreleased_out_of_scope_reservation_allows_new_reservation(self) -> None:
        with self._test_context() as test_context:
            reservation = test_context.instrument._try_reserve_ronchigram_camera()
            reservation = None
            gc.collect()

            reservation2 = test_context.instrument._try_reserve_ronchigram_camera()
            self.assertIsNotNone(reservation2)
            assert reservation2
            self.assertIsNotNone(reservation2.resource)

            reservation2.release()

    def test_cannot_reserve_in_thread(self) -> None:
        with self._test_context() as test_context:
            exception = None
            def reserve_camera():
                nonlocal exception
                try:
                    test_context.instrument._try_reserve_ronchigram_camera()
                except stem_controller.InvalidThreadError as e:
                    exception = e

            thread = threading.Thread(target=reserve_camera)
            thread.start()
            thread.join()
            self.assertIsNotNone(exception)

    def test_can_reserve_in_async(self) -> None:
        with self._test_context() as test_context:
            async def reserve_camera_async():
                with test_context.instrument._try_reserve_ronchigram_camera() as reservation:
                    assert reservation
                    return reservation.resource is not None

            result = asyncio.run(reserve_camera_async())
            self.assertTrue(result)

    def test_available_stream(self) -> None:
        with self._test_context() as test_context:
            self.assertTrue(test_context.instrument.is_ronchigram_available.value)
            with test_context.instrument._try_reserve_ronchigram_camera() as reservation:
                assert reservation
                self.assertIsNotNone(reservation.resource)
                self.assertFalse(test_context.instrument.is_ronchigram_available.value)
            self.assertTrue(test_context.instrument.is_ronchigram_available.value)
