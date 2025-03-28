import typing
import unittest

import numpy

from nion.instrumentation import Acquisition
from nion.instrumentation import AcquisitionLibrary
from nion.instrumentation.test import AcquisitionTestContext
from nion.swift import Application
from nion.swift.test import TestContext
from nion.ui import TestUI
from nion.utils import Geometry


class TestAcquisitionClass(unittest.TestCase):

    def setUp(self) -> None:
        AcquisitionTestContext.begin_leaks()
        self._test_setup = TestContext.TestSetup()

    def tearDown(self) -> None:
        self._test_setup = typing.cast(typing.Any, None)
        AcquisitionTestContext.end_leaks(self)

    def __test_context(self, is_eels: bool=False) -> AcquisitionTestContext.AcquisitionTestContext:
        return AcquisitionTestContext.test_context(is_eels=is_eels, camera_exposure=0.04)

    def test_linear__and_mesh_values(self) -> None:
        values = numpy.array(Acquisition.LinearSpace(start=-1.0, stop=1.0, num=3))
        self.assertEqual((3,), values.shape)
        self.assertTrue(numpy.array_equal(numpy.array([-1.0, 0.0, 1.0], dtype=float), values))
        values = numpy.array(Acquisition.MeshGrid(Acquisition.LinearSpace(start=-1.0, stop=1.0, num=3), Acquisition.LinearSpace(start=-1.5, stop=1.5, num=2)))
        self.assertEqual((3, 2, 2), values.shape)
        self.assertTrue(numpy.array_equal(numpy.array([[[-1.0, -1.5], [-1.0, 1.5]], [[0.0, -1.5], [0.0, 1.5]], [[1.0, -1.5], [1.0, 1.5]]], dtype=float), values))

    def test_single_scan_with_valid_scan_context(self) -> None:
        with self.__test_context() as test_context:
            test_context.scan_hardware_source.scan_context.fov_nm = 100.0
            test_context.scan_hardware_source.scan_context.size = Geometry.IntSize(100, 100)
            test_context.scan_hardware_source.scan_context.rotation_rad = 0.0
            test_context.scan_hardware_source.scan_context.center_nm = Geometry.IntPoint(0, 0)

            acquisition_factory = Acquisition.acquisition_procedure_factory()

            pixel_size: typing.Final = Geometry.IntSize(200, 200)

            scan_device =  acquisition_factory.create_scan_device()
            scan_parameters = acquisition_factory.create_scan_parameters(
                pixel_time_us=1.0,
                pixel_size=pixel_size,
                fov_nm=100,
                rotation_rad=0.0
            )

            device_acquisition_parameters = acquisition_factory.create_device_acquisition_parameters(device=scan_device, device_parameters=scan_parameters, device_channels=(acquisition_factory.create_device_channel_specifier(channel_index=0),))

            scan_acquisition_step = acquisition_factory.create_device_acquisition_step(device_acquisition_parameters=device_acquisition_parameters)

            # define the acquisition procedure using the acquisition step.
            acquisition_procedure = acquisition_factory.create_acquisition_procedure(
                devices=[scan_device],
                steps=(scan_acquisition_step,))

            # import pprint
            # import dataclasses
            # pprint.pprint(dataclasses.asdict(acquisition_procedure))

            # create an acquisition controller. this is the object that will perform the acquisition.
            acquisition_controller = acquisition_factory.create_acquisition_controller(acquisition_procedure=acquisition_procedure)

            # then perform the immediate acquisition.
            acquisition_data = acquisition_controller.acquire_immediate()

            # check the results.
            self.assertEqual(1, len(acquisition_data))
            self.assertEqual((pixel_size.height, pixel_size.width), list(acquisition_data.values())[0].data_shape)

    def test_series_of_ronchigrams(self) -> None:
        with self.__test_context() as test_context:
            acquisition_factory = Acquisition.acquisition_procedure_factory()

            # define the acquisition device (scan) and instrument (stem).
            ronchigram_device = acquisition_factory.create_ronchigram_device()
            stem_device = acquisition_factory.create_stem_device()

            # define the control (with values) to be varied in the table. the values are a 1d array of values.
            # the shape of the array is (3, 1).
            step_size: typing.Final = 1e-9
            controller_1d = acquisition_factory.create_device_controller(
                device=stem_device,
                control_id="defocus",
                values=Acquisition.LinearSpace(start=-step_size, stop=step_size, num=3),
                delay=0.05)

            # define the camera parameters
            ronchigram_parameters = acquisition_factory.create_camera_parameters(exposure_ms=10.0)

            # define the ronchigram acquisition parameters and acquisition step
            ronchigram_acquisition_parameters = acquisition_factory.create_device_acquisition_parameters(device=ronchigram_device, device_parameters=ronchigram_parameters)
            ronchigram_acquisition_step = acquisition_factory.create_device_acquisition_step(device_acquisition_parameters=ronchigram_acquisition_parameters)

            # define the collection step. this generates a series of ronchigrams where each one has a different defocus.
            acquisition_step = acquisition_factory.create_collection_step(
                sub_step=ronchigram_acquisition_step,
                control_controller=controller_1d)

            # define the acquisition procedure using the acquisition step.
            acquisition_procedure = acquisition_factory.create_acquisition_procedure(
                devices=[ronchigram_device, stem_device],
                steps=(acquisition_step,))

            # import pprint
            # import dataclasses
            # pprint.pprint(dataclasses.asdict(acquisition_procedure))

            # create an acquisition controller. this is the object that will perform the acquisition.
            # then perform the immediate acquisition.
            acquisition_controller = acquisition_factory.create_acquisition_controller(acquisition_procedure=acquisition_procedure)
            acquisition_data = acquisition_controller.acquire_immediate()

            # check the results.
            self.assertEqual(1, len(acquisition_data))
            self.assertEqual((3, 1024, 1024), list(acquisition_data.values())[0].data_shape)

    def test_table_of_scans(self) -> None:
        with self.__test_context() as test_context:
            acquisition_factory = Acquisition.acquisition_procedure_factory()

            step_size: typing.Final = 1e-9
            pixel_size: typing.Final = Geometry.IntSize(200, 200)

            # create an acquisition controller. this is the object that will perform the acquisition.
            acquisition_library = AcquisitionLibrary.acquisition_library_factory()

            acquisition_procedure = acquisition_library.create_table_of_scans_acquisition_procedure(
                scan_shape=pixel_size,
                pixel_time_us=1.0,
                fov_nm=100.0,
                control_id="stage_position",
                y_values=Acquisition.LinearSpace(start=-step_size, stop=step_size, num=3),
                x_values=Acquisition.LinearSpace(start=-step_size, stop=step_size, num=3),
                delay=0.05,
                axis_id="correctoraxis",
            )

            # import pprint
            # import dataclasses
            # pprint.pprint(dataclasses.asdict(acquisition_procedure))

            # create an acquisition controller. this is the object that will perform the acquisition.
            acquisition_controller = acquisition_factory.create_acquisition_controller(acquisition_procedure=acquisition_procedure)

            # then perform the immediate acquisition.
            acquisition_data = acquisition_controller.acquire_immediate()

            # check the results.
            self.assertEqual(1, len(acquisition_data))
            self.assertEqual((3, 3, pixel_size.height, pixel_size.width), list(acquisition_data.values())[0].data_shape)

    def test_synchronized_scan_camera(self) -> None:
        with self.__test_context(is_eels=True) as test_context:
            acquisition_factory = Acquisition.acquisition_procedure_factory()

            # define the acquisition device (scan) and camera (eels).
            scan_device = acquisition_factory.create_scan_device()
            eels_device = acquisition_factory.create_eels_device()

            # define the scan context parameters
            context_pixel_size = Geometry.IntSize(180, 240)
            scan_context_parameters = acquisition_factory.create_scan_parameters(
                pixel_time_us=1.0,
                pixel_size=context_pixel_size,
                fov_nm=100,
                rotation_rad=0.0
            )
            scan_acquisition_parameters = acquisition_factory.create_device_acquisition_parameters(device=scan_device,
                                                                                                   device_parameters=scan_context_parameters,
                                                                                                   device_channels=(
                                                                                                   acquisition_factory.create_device_channel_specifier(
                                                                                                       channel_index=0),))

            # define the pre-si acquisition step to record the scan context (typically HAADF)
            pre_scan_acquisition = acquisition_factory.create_device_acquisition_step(device_acquisition_parameters=scan_acquisition_parameters)

            # define the acquisition camera (eels) parameters
            eels_si_parameters = acquisition_factory.create_camera_parameters(exposure_ms=10.0)

            # define the camera (eels 2d) acquisition parameters and acquisition step for the si acquisition.
            # add a processing channel for summing the eels 2d into a eels 1d spectrum.
            sum_processing = acquisition_factory.create_processing_channel(
                processing_id="sum",
                processing_parameters={"axis": 0})
            eels_si_acquisition_parameters = acquisition_factory.create_device_acquisition_parameters(
                device=eels_device,
                device_parameters=eels_si_parameters,
                processing_channels=(sum_processing,))

            # define the si acquisition scan parameters
            si_size: typing.Final = Geometry.IntSize(6, 4)
            scan_si_parameters = acquisition_factory.create_scan_parameters(
                pixel_size=si_size,
                # TODO: subscan parameters
                fov_nm=100,
                rotation_rad=0.0
            )

            # define the scan acquisition parameters and acquisition step for the si acquisition.
            # acquire two channels (typically HAADF and MAADF).
            scan_si_acquisition_parameters = acquisition_factory.create_device_acquisition_parameters(
                device=scan_device,
                device_parameters=scan_si_parameters,
                device_channels=(
                    acquisition_factory.create_device_channel_specifier(channel_index=0),
                    acquisition_factory.create_device_channel_specifier(channel_index=1),
                )
            )

            # define the synchronized scan/eels si acquisition. this is a multi device acquisition step driven by
            # the camera. the scan is a secondary device.
            si_acquisition = acquisition_factory.create_multi_device_acquisition_step(
                primary_device_acquisition_parameters=eels_si_acquisition_parameters,
                secondary_device_acquisition_parameters=[scan_si_acquisition_parameters],
            )

            # define the post-si acquisition step to record the scan context (typically HAADF) using the same
            # parameters as the pre-si acquisition.
            post_scan_acquisition = acquisition_factory.create_device_acquisition_step(device_acquisition_parameters=scan_acquisition_parameters)

            # define the acquisition object.
            acquisition_procedure = acquisition_factory.create_acquisition_procedure(
                devices=[scan_device, eels_device],
                steps=[pre_scan_acquisition, si_acquisition, post_scan_acquisition])

            # import pprint
            # import dataclasses
            # pprint.pprint(dataclasses.asdict(acquisition_procedure))

            # create an acquisition controller. this is the object that will perform the acquisition.
            # then, use the acquisition controller to perform the immediate acquisition.
            acquisition_controller = acquisition_factory.create_acquisition_controller(acquisition_procedure=acquisition_procedure)
            acquisition_data = acquisition_controller.acquire_immediate()

            # check the results.
            self.assertEqual(5, len(acquisition_data))
            self.assertEqual((context_pixel_size.height, context_pixel_size.width), list(acquisition_data.values())[0].data_shape)
            self.assertEqual((si_size.height, si_size.width), list(acquisition_data.values())[1].data_shape)
            self.assertEqual((si_size.height, si_size.width), list(acquisition_data.values())[2].data_shape)
            self.assertEqual((si_size.height, si_size.width, 512), list(acquisition_data.values())[3].data_shape)
            self.assertEqual((context_pixel_size.height, context_pixel_size.width), list(acquisition_data.values())[4].data_shape)

    def test_si_with_drift_correction_within_scan(self) -> None:
        # ensure test fails for case where drift log does not show up due to event loop issue
        # ensure test fails if drift viewer is not updated or updates the wrong number of times

        with self.__test_context(is_eels=True) as test_context:
            acquisition_factory = Acquisition.acquisition_procedure_factory()

            # define the acquisition device (scan) and camera (eels).
            scan_device = acquisition_factory.create_scan_device()
            eels_device = acquisition_factory.create_eels_device()

            # define the acquisition camera (eels) parameters
            eels_si_parameters = acquisition_factory.create_camera_parameters(exposure_ms=10.0)

            # define the camera (eels 2d) acquisition parameters and acquisition step for the si acquisition.
            # add a processing channel for summing the eels 2d into a eels 1d spectrum.
            sum_processing = acquisition_factory.create_processing_channel(
                processing_id="sum",
                processing_parameters={"axis": 0})
            eels_si_acquisition_parameters = acquisition_factory.create_device_acquisition_parameters(
                device=eels_device,
                device_parameters=eels_si_parameters,
                processing_channels=(sum_processing,))

            # define the si acquisition scan parameters
            si_size: typing.Final = Geometry.IntSize(6, 4)
            scan_si_parameters = acquisition_factory.create_scan_parameters(
                pixel_size=si_size,
                # TODO: subscan parameters
                fov_nm=100,
                rotation_rad=0.0
            )

            # define the scan acquisition parameters and acquisition step for the si acquisition.
            # acquire two channels (typically HAADF and MAADF).
            scan_channel_specifier = acquisition_factory.create_device_channel_specifier(channel_index=0)
            scan_si_acquisition_parameters = acquisition_factory.create_device_acquisition_parameters(
                device=scan_device,
                device_parameters=scan_si_parameters,
                device_channels=(scan_channel_specifier,)
            )

            # define the synchronized scan/eels si acquisition. this is a multi device acquisition step driven by
            # the camera. the scan is a secondary device.
            drift_parameters = acquisition_factory.create_drift_parameters(
                drift_correction_enabled=True,
                drift_interval_lines=2,
                drift_channel=scan_channel_specifier,
                drift_region=Geometry.FloatRect.from_tlhw(0.1, 0.7, 0.2, 0.2)
            )
            si_acquisition = acquisition_factory.create_multi_device_acquisition_step(
                primary_device_acquisition_parameters=eels_si_acquisition_parameters,
                secondary_device_acquisition_parameters=[scan_si_acquisition_parameters],
                drift_parameters=drift_parameters
            )

            # define the acquisition object.
            acquisition_procedure = acquisition_factory.create_acquisition_procedure(
                devices=[scan_device, eels_device],
                steps=(si_acquisition,))

            # import pprint
            # import dataclasses
            # pprint.pprint(dataclasses.asdict(acquisition_procedure))

            # create an acquisition controller. this is the object that will perform the acquisition.
            # then, use the acquisition controller to perform the immediate acquisition.
            acquisition_controller = acquisition_factory.create_acquisition_controller(acquisition_procedure=acquisition_procedure)
            acquisition_data = acquisition_controller.acquire_immediate()

            # check the results.
            self.assertEqual(2, len(acquisition_data))
            self.assertEqual((si_size.height, si_size.width), list(acquisition_data.values())[0].data_shape)
            self.assertEqual((si_size.height, si_size.width, 512), list(acquisition_data.values())[1].data_shape)

# TODO: drift correction (using streams to represent drift measurements)
# TODO: drift tracker output when using full scans (scan sequence)
# TODO: test packet processing ordering (threads can complete out of order)
# TODO: 4D STEM
