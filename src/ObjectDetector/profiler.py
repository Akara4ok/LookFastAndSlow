import time

class Record:
    def __init__(self):
        self.total_time = 0
        self.count = 0
        self.it = -1

    def update(self, value, it):
        self.total_time += value
        if(it != self.it):
            self.it = it
            self.count += 1

    def avg(self):
        return self.total_time / self.count

class Profiler:
    def __init__(self, auto_log_time: int = 10):
        self.records = {}
        self.last_time = 0
        self.iteration_start_s = 0
        self.auto_log_time = auto_log_time
        self.last_log_time = time.time()
        self.cur_it = -1

    def iteration_start(self):
        self.iteration_start_s = time.time()
        self.last_time = self.iteration_start_s
        self.cur_it += 1
    
    def iteration_end(self):
        current = time.time()
        value = current - self.iteration_start_s
        self.records["Total"] = self.records.get("Total", Record())
        self.records["Total"].update(value, self.cur_it)
        self.last_time = current

        if(self.auto_log_time is not None):
            cur = time.time()
            if(cur - self.last_log_time > self.auto_log_time):
                self.last_log_time = cur
                self.log()

    def process(self, key: str):
        current = time.time()
        value = current - self.last_time
        self.records[key] = self.records.get(key, Record())
        self.records[key].update(value, self.cur_it)
        self.last_time = current

    def log(self):
        print("=" * 100)
        total_time = self.records["Total"].avg()
        self.log_total(total_time)

        for key, record in self.records.items():
            if(key == "Total"):
                continue
            self.log_single(key, record, total_time)
        print("=" * 100)

    def log_total(self, total_time: int):
        print(f"Total: {total_time:.4f}s")

    def log_single(self, key: str, record: Record, total_time: int):
        avg = record.avg()
        print(f"{key}: {avg:.4f}s {(avg / total_time * 100):.2f}%")