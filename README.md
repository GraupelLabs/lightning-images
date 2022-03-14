# classification
Training scripts for an image classifier based on Pytorch-Lightning

## Activate environment
1. Install [pipenv](https://pipenv.pypa.io/en/latest/install/)
2. Install python packages

```pipenv install```

## Model training

First, add a symbolic link to the dataset, so the training script could find it:

```bash
ln -s /path/to/dataset data/images
```

Note: make sure that `data/images/` now contains folders with classes, e.g.

```
ls -l data/images

- class_1
- class_2
- class_3
...
```


To start the training, type the following

```bash
python training.py data.num_workers=10 data.batch_size=32
```

For more parameters, check [config.yml](config.yaml).

## Model evaluation

```bash
python evaluation.py data.num_workers=10 data.batch_size=32 logging.best_model_path=outputs/2020-10-31/17-15-03/best_model/
```
