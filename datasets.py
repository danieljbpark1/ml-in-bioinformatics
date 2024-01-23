import boto3
import joblib
import s3fs
import torch
from torch.utils.data import Dataset

def load_joblib_from_s3(bucket, object_key):
    s3 = boto3.resource('s3')
    fs = s3fs.S3FileSystem() 
    filename = f"s3://{bucket}/{object_key}"
    with fs.open(filename, encoding='utf8') as fh:
        data = joblib.load(fh)
    return data

class JUND_Dataset(Dataset):
    def __init__(self, s3_bucket, data_dir):
        """Loads X, y, w, a from data_dir."""
        super().__init__()
        
        # load X, y, w, a from data_dir
        X = load_joblib_from_s3(bucket=s3_bucket, object_key=f"{data_dir}/shard-0-X.joblib")
        y = load_joblib_from_s3(bucket=s3_bucket, object_key=f"{data_dir}/shard-0-y.joblib")
        w = load_joblib_from_s3(bucket=s3_bucket, object_key=f"{data_dir}/shard-0-w.joblib")
        a = load_joblib_from_s3(bucket=s3_bucket, object_key=f"{data_dir}/shard-0-a.joblib")
        
        # convert them into torch tensors
        self.X = torch.tensor(data=X, dtype=torch.float32)
        self.y = torch.tensor(data=y, dtype=torch.int8)
        self.w = torch.tensor(data=w, dtype=torch.float32)
        self.a = torch.tensor(data=a, dtype=torch.float32)
        
    def __len__(self):
        """Returns length of dataset."""
        return self.X.size(dim=0)
        
    def __getitem__(self, idx):
        """Returns X, y, w, a values at index idx."""
        return self.X[idx], self.y[idx], self.w[idx], self.a[idx]
