from __future__ import annotations

import typing

import numpy.typing

from nion.instrumentation import Acquisition
from nion.utils import Geometry


class AcquisitionLibrary(typing.Protocol):
    def create_table_of_scans_acquisition_procedure(self, *,
                                                    scan_shape: Geometry.IntSize,
                                                    pixel_time_us: float,
                                                    fov_nm: float,
                                                    control_id: str,
                                                    device_control_id: typing.Optional[str] = None,
                                                    delay: float,
                                                    axis_id: str,
                                                    y_values: numpy.typing.ArrayLike,
                                                    x_values: numpy.typing.ArrayLike) -> Acquisition.AcquisitionProcedureFactoryInterface.AcquisitionProcedure: ...


class AcquisitionLibraryV1(AcquisitionLibrary):
    def create_table_of_scans_acquisition_procedure(self, *,
                                                    scan_shape: Geometry.IntSize,
                                                    pixel_time_us: float,
                                                    fov_nm: float,
                                                    control_id: str,
                                                    device_control_id: typing.Optional[str] = None,
                                                    delay: float,
                                                    axis_id: str,
                                                    y_values: numpy.typing.ArrayLike,
                                                    x_values: numpy.typing.ArrayLike) -> Acquisition.AcquisitionProcedureFactoryInterface.AcquisitionProcedure:
        acquisition_factory = Acquisition.acquisition_procedure_factory()

        # define the acquisition device (scan) and instrument (stem).
        scan_device = acquisition_factory.create_scan_device()
        stem_device = acquisition_factory.create_stem_device()

        # define the control to be varied in the table of scans.
        controller_2d = acquisition_factory.create_device_controller(
            device=stem_device,
            control_id=control_id,
            device_control_id=device_control_id,
            values=Acquisition.MeshGrid(y_values, x_values),
            delay=delay,
            axis_id=axis_id)

        # define the scan context parameters
        scan_parameters = acquisition_factory.create_scan_parameters(
            pixel_time_us=pixel_time_us,
            pixel_size=scan_shape,
            fov_nm=fov_nm
        )

        # define the scan acquisition parameters and acquisition step.
        scan_acquisition_parameters = acquisition_factory.create_device_acquisition_parameters(device=scan_device,
                                                                                               device_parameters=scan_parameters)
        scan_acquisition_step = acquisition_factory.create_device_acquisition_step(
            device_acquisition_parameters=scan_acquisition_parameters)

        # define the collection step. this generates a table of scans where each one has a different stage position.
        acquisition_step = acquisition_factory.create_collection_step(
            sub_step=scan_acquisition_step,
            control_controller=controller_2d)

        # define the acquisition procedure using the acquisition step.
        return acquisition_factory.create_acquisition_procedure(
            devices=[scan_device, stem_device],
            steps=(acquisition_step,))


def acquisition_library_factory() -> AcquisitionLibrary:
    return AcquisitionLibraryV1()
