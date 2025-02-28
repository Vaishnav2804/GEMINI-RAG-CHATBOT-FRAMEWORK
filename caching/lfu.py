from collections import defaultdict, OrderedDict

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = {}  # session_id -> (value, freq)
        self.freq_map = defaultdict(OrderedDict)  # freq -> {session_id: None}
        self.min_freq = 0

    def _update_freq(self, session_id):
        value, freq = self.data[session_id]
        del self.freq_map[freq][session_id]
        if not self.freq_map[freq]:
            del self.freq_map[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        
        new_freq = freq + 1
        self.data[session_id] = (value, new_freq)
        self.freq_map[new_freq][session_id] = None

    def get(self, session_id):
        if session_id not in self.data:
            return None
        self._update_freq(session_id)
        return self.data[session_id][0]

    def put(self, session_id, value):
        if self.capacity == 0:
            return
        
        if session_id in self.data:
            self.data[session_id] = (value, self.data[session_id][1])
            self._update_freq(session_id)
        else:
            if len(self.data) >= self.capacity:
                # Evict the least frequently used item
                lfu_session_id, _ = self.freq_map[self.min_freq].popitem(last=False)
                del self.data[lfu_session_id]
                
            self.data[session_id] = (value, 1)
            self.freq_map[1][session_id] = None
            self.min_freq = 1