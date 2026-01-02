import threading
from queue import Empty, Queue

import matplotlib.pyplot as plt


class DebugDisplay:
    """
    Safely displays a single interactive image in the main thread.
    - Ensures plt.show() and plt.close() are called in main thread.
    - Queues new images to replace the previous one.
    - Blocks execution until the user closes the window.
    """

    def __init__(self):
        self._image_queue = Queue(maxsize=1)
        self._lock = threading.Lock()
        self._figure = None
        self._ax = None

    def show_image(self, img, title="Debug Image"):
        """
        Queue an image to display. Replaces any previous queued image.
        """
        with self._lock:
            if not self._image_queue.empty():
                try:
                    self._image_queue.get_nowait()
                except Empty:
                    pass
            self._image_queue.put((img, title))
            # Directly display if not already showing
            if self._figure is None:
                self._display_main_thread()

    def _display_main_thread(self):
        # Only call this in main thread
        while not self._image_queue.empty():
            img, title = self._image_queue.get()
            self._figure, self._ax = plt.subplots()
            self._ax.imshow(img)
            self._ax.set_title(title)
            plt.show(block=True)  # blocks until user closes
            plt.close(self._figure)
            self._figure = None
            self._ax = None

    def close(self):
        """
        Manually close the window if still open.
        Safe to call multiple times.
        """
        with self._lock:
            if self._figure is not None:
                plt.close(self._figure)
                self._figure = None
                self._ax = None
