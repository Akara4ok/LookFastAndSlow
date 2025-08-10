import torch

class CachedDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.cached_batches = []
        self._cached = False

    def __iter__(self):
        if self._cached:
            for batch in self.cached_batches:
                yield batch
        else:
            for batch in self.dataloader:
                self.cached_batches.append([
                    x.detach().clone() if isinstance(x, torch.Tensor) else x
                    for x in batch
                ])
                yield self.cached_batches[-1]
            self._cached = True

    def __len__(self):
        return len(self.cached_batches) if self._cached else len(self.dataloader)