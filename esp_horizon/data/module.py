import os
import pytorch_lightning as pl
from ptls.frames import PtlsDataModule
from .dataset import ESPDataset


def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        if d is not None:
            result.update(d)
    return result


class ESPDataModule(PtlsDataModule):
    def __init__(self,
                 train_path=None,
                 train_batch_size=1,
                 train_num_workers=0,
                 train_params=None,
                 dev_path=None,
                 dev_batch_size=None,
                 dev_num_workers=None,
                 dev_params=None,
                 test_path=None,
                 test_batch_size=None,
                 test_num_workers=None,
                 test_params=None,
                 **params
                 ):
        if train_path is not None:
            train_data = ESPDataset(train_path, **merge_dicts(params, train_params))
        else:
            train_data = None
        if dev_path is not None:
            dev_data = ESPDataset(dev_path, **merge_dicts(params, dev_params))
        else:
            dev_data = None
        if test_path is not None:
            test_data = ESPDataset(test_path, **merge_dicts(params, test_params))
        else:
            test_data = None

        super().__init__(
            train_data=train_data,
            train_batch_size=train_batch_size,
            train_num_workers=train_num_workers,
            train_drop_last=True,
            valid_data=dev_data,
            valid_batch_size=dev_batch_size or train_batch_size,
            valid_num_workers=dev_num_workers or train_num_workers,
            valid_drop_last=False,
            test_data=test_data,
            test_batch_size=test_batch_size or dev_batch_size or train_batch_size,
            test_num_workers=test_num_workers or dev_num_workers or train_num_workers,
            test_drop_last=False
        )

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        train_id_field = train_data.id_field if train_data is not None else None
        dev_id_field = dev_data.id_field if dev_data is not None else None
        test_id_field = test_data.id_field if test_data is not None else None
        id_field = train_id_field or dev_id_field or test_id_field
        if ((train_id_field and (train_id_field != id_field)) or
            (dev_id_field and (dev_id_field != id_field)) or
            (test_id_field and (test_id_field != id_field))):
            raise ValueError("Different id fields in data splits.")
        if id_field is None:
            raise ValueError("No datasets provided.")
        self.id_field = id_field
