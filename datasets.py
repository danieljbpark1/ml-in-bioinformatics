import boto3
import joblib
import s3fs
import torch
from torch.utils.data import Dataset

def load_joblib_from_s3(
    s3_dir: str, 
    filename: str
):
    """Loads a .joblib file from an S3 Bucket.

    Args:
        s3_dir (str): Directory in S3 Bucket.
        filename (str): Filename in `s3_dir`.

    Returns:
        Any: A Python object reconstructed by joblib.
    """
    assert s3_dir.startswith("s3://")

    fs = s3fs.S3FileSystem() 
    s3_object_key = f"{s3_dir}/{filename}"

    with fs.open(s3_object_key, encoding='utf8') as fh:
        data = joblib.load(fh)

    return data

class JUND_Dataset(Dataset):
    """JUND transcription factor Dataset."""

    def __init__(self, data_dir: str):
        """Initializes instance of class JUND_Dataset.
        
        Args:
            data_dir (str): Path to directory containing joblib files.
        """
        super().__init__()
        
        if data_dir.startswith("s3://"):
            X = load_joblib_from_s3(s3_dir=data_dir, filename="shard-0-X.joblib")
            y = load_joblib_from_s3(s3_dir=data_dir, filename="shard-0-y.joblib")
            w = load_joblib_from_s3(s3_dir=data_dir, filename="shard-0-w.joblib")
            a = load_joblib_from_s3(s3_dir=data_dir, filename="shard-0-a.joblib")
        else:
            X = joblib.load(filename=f"{data_dir}/shard-0-X.joblib")
            y = joblib.load(filename=f"{data_dir}/shard-0-y.joblib")
            w = joblib.load(filename=f"{data_dir}/shard-0-w.joblib")
            a = joblib.load(filename=f"{data_dir}/shard-0-a.joblib")

        # convert them into torch tensors
        self.X = torch.tensor(data=X, dtype=torch.float32)
        self.y = torch.tensor(data=y, dtype=torch.float32)
        self.w = torch.tensor(data=w, dtype=torch.float32)
        self.a = torch.tensor(data=a, dtype=torch.float32)
        
    def __len__(self):
        """Returns length of dataset."""
        return self.X.size(dim=0)
        
    def __getitem__(self, idx):
        """Returns X, y, w, a Tensors at index idx."""
        return self.X[idx], self.y[idx], self.w[idx], self.a[idx]
