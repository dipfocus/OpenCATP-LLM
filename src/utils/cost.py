import os
import time
from threading import Thread, Event

import psutil

from src.config import GlobalMetricsConfig


class CPUMemoryMonitor(Thread):
    """
    后台线程，每隔 interval 秒抓取一次指定进程的内存占用，记录最大值。
    """
    pid: int
    interval: float
    max_memory: int
    _stop_event: Event

    def __init__(self, *, pid=None, interval=0.1):
        super().__init__()
        self.pid = pid or os.getpid()
        self.interval = interval
        self.max_memory = 0
        self._stop_event = Event()

    def run(self):
        process = psutil.Process(self.pid)
        while not self._stop_event.is_set():
            try:
                mem_info = process.memory_info()
                # RSS (Resident Set Size)，单位是字节
                rss = mem_info.rss
                if rss > self.max_memory:
                    self.max_memory = rss
            except psutil.NoSuchProcess:
                # 如果进程不存在了，就退出循环
                break
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join()

    def get_max_cpu_memory_allocated(self, *, size='MB'):
        if size == 'MB':
            return self.max_memory / (1024 ** 2)


def calculate_qop(
        score,
        cost,
        alpha=GlobalMetricsConfig.ALPHA,
        min_score=GlobalMetricsConfig.MIN_SCORE,
        max_score=GlobalMetricsConfig.MAX_SCORE,
        min_cost=GlobalMetricsConfig.MIN_COST,
        max_cost=GlobalMetricsConfig.MAX_COST
):
    norm_score = (score - min_score) / (max_score - min_score)
    norm_cost = (cost - min_cost) / (max_cost - min_cost)
    qop = alpha * norm_score - (1 - alpha) * norm_cost
    return qop
