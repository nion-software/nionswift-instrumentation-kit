# a set of classes to make observing lists easier. these will eventually be moved to nionutils.

from __future__ import annotations

# system imports
import typing
import weakref

# local imports
from nion.utils import Event
from nion.utils import ListModel
from nion.utils.ReferenceCounting import weak_partial


class ListItemHandlerLike(typing.Protocol):
    def begin(self, item_stack: typing.Sequence[typing.Any]) -> None: ...
    def end(self) -> None: ...


class ListItemHandlerFactoryLike(typing.Protocol):
    def make(self) -> ListItemHandlerLike: ...


class ListItemEventsHandler(ListItemHandlerLike):
    def __init__(self, event_map: typing.Mapping[str, Event.EventListenerCallableType]) -> None:
        self.__listeners: typing.Dict[str, Event.EventListener] = dict()
        self.__event_map = event_map

    def begin(self, item_stack: typing.Sequence[typing.Any]) -> None:
        for event_name in self.__event_map.keys():
            item = item_stack[-1]
            self.__listeners[event_name] = getattr(item, event_name).listen(
                weak_partial(ListItemEventsHandler.__bounce, self, event_name, [weakref.ref(i) for i in item_stack]))

    def end(self) -> None:
        pass

    def __bounce(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        event_name = args[0]
        item_stack = [a() for a in args[1]]
        if all(item_stack):
            self.__event_map[event_name](*item_stack, *args[2:], **kwargs)


class ListItemEventsHandlerFactory(ListItemHandlerFactoryLike):
    def __init__(self, event_map: typing.Mapping[str, Event.EventListenerCallableType]) -> None:
        self.__event_map = event_map

    def make(self) -> ListItemHandlerLike:
        return ListItemEventsHandler(self.__event_map)


class ListItemItemsHandlerBase(ListItemHandlerLike):
    def __init__(self, factory: ListItemHandlerFactoryLike, *, key: typing.Optional[str] = None) -> None:
        self.__factory = factory
        self.__key = key or "items"
        self.__item_handlers: typing.List[ListItemHandlerLike] = list()

    def begin(self, item_stack: typing.Sequence[typing.Any]) -> None:
        raise NotImplementedError()

    def end(self) -> None:
        pass

    def _begin(self, list_model: ListModel.ListModel[typing.Any], item_stack: typing.Sequence[typing.Any]) -> None:
        def item_inserted(key: str, item: typing.Any, before_index: int) -> None:
            if key == self.__key:
                a = self.__factory.make()
                self.__item_handlers.insert(before_index, a)
                a.begin(list(item_stack) + [item])

        def item_removed(key: str, item: typing.Any, index: int) -> None:
            if key == self.__key:
                self.__item_handlers[index].end()
                del self.__item_handlers[index]

        self.__item_inserted_listener = list_model.item_inserted_event.listen(item_inserted)
        self.__item_removed_listener = list_model.item_removed_event.listen(item_removed)

        for i in range(len(list_model.items)):
            item_inserted(self.__key, list_model.items[i], i)


class ListItemItemsHandler(ListItemItemsHandlerBase):
    def __init__(self, list_key: str, factory: ListItemHandlerFactoryLike, *, key: typing.Optional[str] = None) -> None:
        super().__init__(factory, key=key)
        self.__list_key = list_key

    def begin(self, item_stack: typing.Sequence[typing.Any]) -> None:
        item = item_stack[-1]
        self._begin(getattr(item, self.__list_key), item_stack)

    def end(self) -> None:
        pass


class ListItemItemsHandlerFactory(ListItemHandlerFactoryLike):
    def __init__(self, list_key: str, factory: ListItemHandlerFactoryLike, *, key: typing.Optional[str] = None) -> None:
        self.__list_key = list_key
        self.__factory = factory
        self.__key = key

    def make(self) -> ListItemHandlerLike:
        return ListItemItemsHandler(self.__list_key, self.__factory, key=self.__key)


class ListListener(ListItemItemsHandlerBase):
    """Listen to events on a list model and send them to listeners.

    The list listener watches for items inserted/removed from the model and adds listeners based on the mapping
    of event names to functions; the function will be called with the inserted/removed item as the first parameter.
    """
    def __init__(self, list_model: ListModel.ListModel[typing.Any], factory: ListItemHandlerFactoryLike, *, key: typing.Optional[str] = None) -> None:
        super().__init__(factory, key=key)
        self._begin(list_model, list())
