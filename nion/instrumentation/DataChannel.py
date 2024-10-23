from __future__ import annotations

# system imports
import numpy
import numpy.typing
import typing
import weakref

# local libraries
from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition
from nion.swift.model import ApplicationData
from nion.swift.model import DataItem
from nion.swift.model import DocumentModel

_NDArray = numpy.typing.NDArray[typing.Any]


class DataItemReference:
    def __init__(self, document_model: DocumentModel.DocumentModel, data_item: DataItem.DataItem) -> None:
        self.__document_model = document_model
        self.__data_item = data_item
        self.__data_item.increment_data_ref_count()
        self.__data_item_transaction = document_model.item_transaction(self.__data_item)
        self.__document_model.begin_data_item_live(self.__data_item)

        def finalize(data_item_transaction: DocumentModel.Transaction) -> None:
            data_item_transaction.close()
            document_model.end_data_item_live(data_item)
            data_item.decrement_data_ref_count()

        weakref.finalize(self, finalize, self.__data_item_transaction)

    @property
    def data_item(self) -> DataItem.DataItem:
        return self.__data_item


class DataItemDataChannel(Acquisition.DataChannel):
    """Acquisition data channel backed by a data item.

    An acquisition data channel receives partial data and must return full data when required.
    """

    def __init__(self, document_model: DocumentModel.DocumentModel, title_base: str, channel_names: typing.Mapping[Acquisition.Channel, str]):
        super().__init__()
        self.__document_model = document_model
        self.__title_base = title_base
        self.__channel_names = dict(channel_names)
        self.__data_item_ref_map: typing.Dict[Acquisition.Channel, DataItemReference] = dict()
        self.on_display_data_item: typing.Optional[typing.Callable[[DataItem.DataItem], None]] = None

    def prepare(self, channel_info_map: typing.Mapping[Acquisition.Channel, Acquisition.DataStreamInfo]) -> None:
        # prepare will be called on the main thread.
        acquisition_number = Acquisition.session_manager.get_project_acquisition_index(self.__document_model)
        for channel, data_stream_info in channel_info_map.items():
            if self.__title_base:
                title = f"{self.__title_base} {self.__channel_names.get(channel, str(channel))} {acquisition_number}"
            else:
                title = f"{self.__channel_names.get(channel, str(channel))} {acquisition_number}"
            data_item = self.__create_data_item(data_stream_info.data_metadata, title)
            self.__data_item_ref_map[channel] = DataItemReference(self.__document_model, data_item)
            if callable(self.on_display_data_item):
                self.on_display_data_item(data_item)

    def get_data(self, channel: Acquisition.Channel) -> DataAndMetadata.DataAndMetadata:
        data_and_metadata = self.__data_item_ref_map[channel].data_item.data_and_metadata
        assert data_and_metadata
        return data_and_metadata

    def get_data_item(self, channel: Acquisition.Channel) -> DataItem.DataItem:
        return self.__data_item_ref_map[channel].data_item

    def update_data(self, channel: Acquisition.Channel, source_data: _NDArray, source_slice: Acquisition.SliceType, dest_slice: slice, data_metadata: DataAndMetadata.DataMetadata) -> None:
        data_item_ref = self.__data_item_ref_map.get(channel, None)
        if data_item_ref:
            source_data_and_metadata = DataAndMetadata.new_data_and_metadata(source_data, data_descriptor=data_metadata.data_descriptor)
            dest_slice_lists = Acquisition.simple_unravel_flat_slice(dest_slice, data_metadata.data_shape)
            assert len(dest_slice_lists) == 1  # otherwise we need to break up the source slices too. skipping until needed.
            for dest_slices in dest_slice_lists:
                self.__document_model.update_data_item_partial(data_item_ref.data_item, data_metadata, source_data_and_metadata, source_slice, dest_slices)

    def __create_data_item(self, data_metadata: DataAndMetadata.DataMetadata, title: str) -> DataItem.DataItem:
        data_shape = data_metadata.data_shape
        data_descriptor = data_metadata.data_descriptor
        large_format = bool(numpy.prod(data_shape, dtype=numpy.int64) > 2048**2 * 10)
        data_item = DataItem.DataItem(large_format=large_format)
        data_item.title = title
        self.__document_model.append_data_item(data_item)
        data_item.reserve_data(data_shape=data_shape, data_dtype=numpy.dtype(numpy.float32), data_descriptor=data_descriptor)
        session_metadata_dict = ApplicationData.get_session_metadata_dict()
        Acquisition.session_manager.update_session_metadata_dict(self.__document_model, session_metadata_dict)
        data_item.session_metadata = session_metadata_dict
        return data_item
