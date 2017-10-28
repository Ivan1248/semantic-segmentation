class Event(object):
    """
    Standard event
        Ways to declare an event:
            @Event
            def click(self, x, y): pass  # decorated dummy method
            self.click = Event()         # or instance variable in __init__ method
        Adding/removing event handlers:
            canvas.click += shape.clone  # canvas.click._handlers.add(shape.clone)
            canvas.click -= shape.clone  # canvas.click._handlers.remove(shape.clone)
        Raising event:
            self.click(x,y)              # for h in self.click._handlers: h(x,y)

    Event with response
        Declaring an event that returns the value from its handler:
            @Event
            def closing(self):
                cancel = False
                return cancel           # the default return value
        Adding event handlers:
            window.closing += lambda: not self.ready_to_close_window
        Raising event:
            cancel = self.closing()     # The response will be equal to the return-value
            if cancel:                  # of the first handler that doesn't return or the
                return                  # default return value i all handlers return None.
            else:
                self.close()
    """
    def __init__(self, f=lambda: None):
        self._handlers = list()
        self._default_response = f()

    def __call__(self, *args, **kwargs):
        value = self._default_response
        for h in self._handlers:
            value = h(*args, **kwargs) or value
        return value
    notify = __call__

    def __add__(self, handler):
        if handler not in self._handlers:
            self._handlers.append(handler)
        return self
    add_handler = __add__

    def __isub__(self, handler):
        if handler in self._handlers:
            self._handlers.remove(handler)
        return self
    remove_handler = __isub__

    def __len__(self):
        return len(self._handlers)

    def __repr__(self):
        return "Event({}, {})".format(self._handlers, self._default_response)


class HandlesEvent(object):
    def __init__(self, f, event):
        self
        pass

    def __call__(self, f):
        def wrapped_f(*args):
            f(*args)
        return wrapped_f

"""class EventHandler(object):
    def __init__(self, delegate, event):
        self.handlers = set()
        self.f = f

    def __call__(self, *args):
        self.f()"""


class EventTestClass1(object):
    def __init__(self):
        self.some_event += self.some_event_handler
        self.some_event("blabla")
        pass

    @Event()
    def some_event(self, message): pass

    @HandlesEvent('self.some_event')
    def some_event_handler(self, message):
        print(message)


testobj = EventTestClass1()
