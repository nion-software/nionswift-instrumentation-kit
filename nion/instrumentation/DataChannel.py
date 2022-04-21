from __future__ import annotations

# system imports
import numpy
import numpy.typing
import typing

# local libraries
from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition
from nion.swift.model import ApplicationData
from nion.swift.model import DataItem
from nion.swift.model import DocumentModel

_NDArray = numpy.typing.NDArray[typing.Any]


class DataItemDataChannel(Acquisition.DataChannel):
    """Acquisition data channel backed by a data item.

    An acquisition data channel receives partial data and must return full data when required.
    """

    def __init__(self, document_model: DocumentModel.DocumentModel, title_base: str, channel_names: typing.Mapping[Acquisition.Channel, str]):
        super().__init__()
        self.__document_model = document_model
        self.__title_base = title_base
        self.__channel_names = dict(channel_names)
        self.__data_item_map: typing.Dict[Acquisition.Channel, DataItem.DataItem] = dict()
        self.__data_item_transaction_map: typing.Dict[Acquisition.Channel, DocumentModel.Transaction] = dict()
        self.on_display_data_item: typing.Optional[typing.Callable[[DataItem.DataItem], None]] = None

    def prepare(self, data_stream: Acquisition.DataStream) -> None:
        # prepare will be called on the main thread.
        for channel in data_stream.channels:
            data_stream_info = data_stream.get_info(channel)
            if self.__title_base:
                title = f"{self.__title_base} {self.__channel_names.get(channel, str(channel))}"
            else:
                title = f"{self.__channel_names.get(channel, str(channel))}"
            data_item = self.__create_data_item(data_stream_info.data_metadata, title)
            self.__data_item_map[channel] = data_item
            data_item.increment_data_ref_count()
            self.__data_item_transaction_map[channel] = self.__document_model.item_transaction(data_item)
            self.__document_model.begin_data_item_live(data_item)
            if callable(self.on_display_data_item):
                self.on_display_data_item(data_item)

    def about_to_delete(self) -> None:
        for channel in self.__data_item_map.keys():
            data_item = self.__data_item_map[channel]
            data_item_transaction = self.__data_item_transaction_map[channel]
            data_item_transaction.close()
            self.__document_model.end_data_item_live(data_item)
            data_item.decrement_data_ref_count()
        self.__data_item_map.clear()
        self.__data_item_transaction_map.clear()
        super().about_to_delete()

    def get_data(self, channel: Acquisition.Channel) -> DataAndMetadata.DataAndMetadata:
        data_and_metadata = self.__data_item_map[channel].data_and_metadata
        assert data_and_metadata
        return data_and_metadata

    def get_data_item(self, channel: Acquisition.Channel) -> DataItem.DataItem:
        return self.__data_item_map[channel]

    def update_data(self, channel: Acquisition.Channel, source_data: _NDArray, source_slice: Acquisition.SliceType, dest_slice: slice, data_metadata: DataAndMetadata.DataMetadata) -> None:
        data_item = self.__data_item_map.get(channel, None)
        if data_item:
            source_data_and_metadata = DataAndMetadata.new_data_and_metadata(source_data, data_descriptor=data_metadata.data_descriptor)
            dest_slice_lists = Acquisition.unravel_flat_slice(dest_slice, data_metadata.data_shape)
            assert len(dest_slice_lists) == 1  # otherwise we need to break up the source slices too. skipping until needed.
            for dest_slices in dest_slice_lists:
                self.__document_model.update_data_item_partial(data_item, data_metadata, source_data_and_metadata, source_slice, dest_slices)

    def __create_data_item(self, data_metadata: DataAndMetadata.DataMetadata, title: str) -> DataItem.DataItem:
        data_shape = data_metadata.data_shape
        data_descriptor = data_metadata.data_descriptor
        large_format = bool(numpy.prod(data_shape, dtype=numpy.int64) > 2048**2 * 10)
        data_item = DataItem.DataItem(large_format=large_format)
        data_item.title = title
        self.__document_model.append_data_item(data_item)
        data_item.reserve_data(data_shape=data_shape, data_dtype=numpy.dtype(numpy.float32), data_descriptor=data_descriptor)
        data_item.session_metadata = ApplicationData.get_session_metadata_dict()
        return data_item
