import abc
from queue import Empty, Queue
from threading import Event, Thread
from threading import enumerate as threading_enumerate

from dolphin._log import get_log

logger = get_log()

_DEFAULT_TIMEOUT = 1.0


def is_main_thread_active() -> bool:
    """Check if the main thread is still active.

    Used to check if the writing thread should exit if there was
    some exception in the main thread.

    Source: https://stackoverflow.com/a/23443397/4174466
    """
    return any((i.name == "MainThread") and i.is_alive() for i in threading_enumerate())


class BackgroundWorker(abc.ABC):
    """Base class for doing work in a background thread.

    After instantiating an object, a client sends it work with the `queue_work`
    method and retrieves the result with the `get_result` method (hopefully
    after doing something else useful in between).  The worker remains active
    until `notify_finished` is called.  Subclasses must define the `process`
    method.

    Parameters
    ----------
    num_work_queue : int
        Max number of work items to queue before blocking, <= 0 for unbounded.
    num_results_queue : int
        Max number of results to generate before blocking, <= 0 for unbounded.
    store_results : bool
        Whether to store return values of `process` method.  If True then
        `get_result` must be called once for every `queue_work` call.
    timeout : float
        Interval in seconds used to check for finished notification once work
        queue is empty.

    Notes
    -----
    The usual caveats about Python threading apply.  It's typically a poor
    choice for concurrency unless the global interpreter lock (GIL) has been
    released, which can happen in IO calls and compiled extensions.
    """

    def __init__(
        self,
        num_work_queue=0,
        num_results_queue=0,
        store_results=True,
        timeout=_DEFAULT_TIMEOUT,
    ):
        self.store_results = store_results
        self.timeout = timeout
        self._finished_event = Event()
        self._work_queue = Queue(num_work_queue)
        self._results_queue = Queue(num_results_queue)
        self._thread = Thread(target=self._consume_work_queue)
        self._thread.start()

    def _consume_work_queue(self):
        while not self._finished_event.is_set():
            logger.debug("getting work")
            try:
                args, kw = self._work_queue.get(timeout=self.timeout)
            except Empty:
                logger.debug("timed out, checking if done")
                continue
            logger.debug("processing")
            result = self.process(*args, **kw)
            logger.debug("got result")
            if self.store_results:
                logger.debug("saving result in queue")
                self._results_queue.put(result)

    @abc.abstractmethod
    def process(self, *args, **kw):
        """User-defined task to operate in background thread."""
        pass

    def queue_work(self, *args, **kw):
        """Add a job to the work queue to be executed.

        Blocks if work queue is full.
        Same input interface as `process`.
        """
        if self._finished_event.is_set():
            raise RuntimeError("Attempted to queue_work after notify_finished!")
        self._work_queue.put((args, kw))

    def get_result(self):
        """Get the least-recent value from the result queue.

        Blocks until a result is available.
        Same output interface as `process`.
        """
        result = self._results_queue.get()
        self._results_queue.task_done()
        return result

    def notify_finished(self):
        """Signal that all work has finished.

        Indicate that no more work will be added to the queue, and block until
        all work has been processed.
        If `store_results=True` also block until all results have been retrieved.
        """
        self._finished_event.set()
        self._results_queue.join()
        self._thread.join()

    def __del__(self):
        self.notify_finished()


class BackgroundWriter(BackgroundWorker):
    """Base class for writing data in a background thread.

    After instantiating an object, a client sends it data with the `queue_write`
    method.  The writer remains active until `notify_finished` is called.
    Subclasses must define the `write` method.

    Parameters
    ----------
    nq : int
        Number of write jobs that can be queued before blocking, <= 0 for
        unbounded.  Default is 1.
    timeout : float
        Interval in seconds used to check for finished notification once write
        queue is empty.
    """

    def __init__(self, nq=1, timeout=_DEFAULT_TIMEOUT):
        super().__init__(num_work_queue=nq, store_results=False, timeout=timeout)

    # rename queue_work -> queue_write
    def queue_write(self, *args, **kw):
        """Add data to the queue to be written.

        Blocks if write queue is full.
        Same interfaces as `write`.
        """
        self.queue_work(*args, **kw)

    # rename process -> write
    def process(self, *args, **kw):
        self.write(*args, **kw)

    @abc.abstractmethod
    def write(self, *args, **kw):
        """User-defined method for writing data."""
        pass


class BackgroundReader(BackgroundWorker):
    """Base class for reading data in a background thread (pre-fetching).

    After instantiating an object, a client sends it data selection parameters
    (slices, indices, etc.) via the `queue_read` method and retrives the result
    with the `get_data` method.  In order to get useful concurrency, that
    usually means you'll want to queue the read for the next data block before
    starting work on the current block.  The reader remains active until
    `notify_finished` is called and all blocks have been retrieved.  Subclasses
    must define the `read` method.

    Parameters
    ----------
    nq : int
        Number of read results that can be stored before blocking, <= 0 for
        unbounded.  Default is 1.
    timeout : float
        Interval in seconds used to check for finished notification once write
        queue is empty.
    """

    def __init__(self, nq=1, timeout=_DEFAULT_TIMEOUT):
        super().__init__(num_results_queue=nq, timeout=timeout)

    # rename queue_work -> queue_read
    def queue_read(self, *args, **kw):
        """Add selection parameters (slices, etc.) to the read queue to be processed.

        Same input interface as `read`.
        """
        self.queue_work(*args, **kw)

    # rename get_result -> get_data
    def get_data(self):
        """Retrieve the least-recently read chunk of data.

        Blocks until a result is available.
        Same output interface as `read`.
        """
        return self.get_result()

    # rename process -> read
    def process(self, *args, **kw):
        return self.read(*args, **kw)

    @abc.abstractmethod
    def read(self, *args, **kw):
        """User-defined method for reading a chunk of data."""
        pass