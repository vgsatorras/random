import torch
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.loader import TrainDataLoader
from gluonts.torch.batchify import batchify
from gluonts.itertools import Cached
from gluonts.dataset.field_names import FieldName
from gluonts.transform import AddObservedValuesIndicator, InstanceSplitter, ExpectedNumInstanceSampler
import time
import numpy as np

max_num_nodes = None
batch_size = 32
num_batches_per_epoch = 20
dataset_name = "electricity"
num_workers = 1
context_length = 12
pred_length = 12


def get_dataloader(name):
    dataset = get_dataset(name, regenerate=False)
    grouper_train = MultivariateGrouper(max_num_nodes)

    mv_dataset = TrainDatasets(
        metadata=dataset.metadata,
        train=grouper_train(dataset.train),
    )

    transformation = get_mask_trans() + get_train_split_trans(pred_length, context_length)
    data_loader = TrainDataLoader(Cached(mv_dataset.train), batch_size=batch_size, stack_fn=batchify, transform=transformation,
                                  num_batches_per_epoch=num_batches_per_epoch, num_workers=num_workers)

    return data_loader


def get_train_split_trans(prediction_length, context_length):
    training_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExpectedNumInstanceSampler(
            num_instances=1,
            min_future=prediction_length,
        ),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )
    return training_splitter


def get_mask_trans():
    # Replaces nans in the target field with a dummy value (zero), and adds a field indicating which values were actually observed vs imputed this way.
    mask_unobserved = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    )
    return mask_unobserved


class SimpleDataloader:
    def __init__(self, num_nodes, series_length, context_length, pred_length):
        self.data = torch.randn(num_nodes, series_length)
        self.context_length = context_length
        self.pred_length = pred_length
        self.series_length = series_length

    def sample(self, batch_size):
        positions = np.random.randint(self.context_length, self.series_length - self.pred_length, batch_size).astype(int)
        batch = {"past": [], "future": []}
        for pos in positions:
            past = self.data[:, (pos-self.context_length):pos].unsqueeze(0)
            future = self.data[:,pos:(pos + self.pred_length)].unsqueeze(0)
            batch["past"].append(past)
            batch["future"].append(future)
        batch["past"] = torch.cat(batch["past"])
        batch["future"] = torch.cat(batch["future"])
        return batch


def assess_time():
    gt_dataloader = get_dataloader(dataset_name)
    counter = 0
    t1 = time.time()
    for batch in gt_dataloader:
        counter += 1
    t2 = time.time()
    gt_avg_time = (t2 - t1)/counter
    print("Average time GluonTS dataloader = %.4f" % gt_avg_time)


    num_nodes = 321  # Number of nodes for electricity time series
    series_length = 26304  # maximum length of electricity time series
    basic_dataloader = SimpleDataloader(num_nodes, series_length, context_length, pred_length)
    t3 = time.time()
    for i in range(counter):
        batch = basic_dataloader.sample(batch_size)
    t4 = time.time()
    gt_avg_time = (t4 - t3)/counter
    print("Average time Basic dataloader = %.4f" % gt_avg_time)


if __name__ == '__main__':
    assess_time()
