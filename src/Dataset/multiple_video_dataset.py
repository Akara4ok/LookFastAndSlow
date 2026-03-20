from pathlib import Path
from torch.utils.data import Dataset
from Dataset.single_video_dataset import SingleVideoDataset

class MultipleVideoDataset(Dataset):
    def __init__(self, root: str, seq_len: int = 6):
        self.root = Path(root)
        self.seq_len = seq_len

        self.video_datasets = []
        self.index_map = []  # [(dataset_idx, local_idx), ...]

        for subdir in sorted(self.root.iterdir()):
            if not subdir.is_dir():
                continue

            video_file = subdir / "video.mp4"
            split_file = subdir / "ImageSets" / "Main" / "Train.txt"

            if not video_file.exists() or not split_file.exists():
                continue

            ds = SingleVideoDataset(str(subdir), seq_len=seq_len)

            num_sequences = len(ds)
            if num_sequences == 0:
                continue

            dataset_idx = len(self.video_datasets)
            self.video_datasets.append(ds)

            for local_idx in range(num_sequences):
                self.index_map.append((dataset_idx, local_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_idx, local_idx = self.index_map[idx]
        return self.video_datasets[dataset_idx][local_idx]