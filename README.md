# classification
Training scripts for an image classifier based on Pytorch-Lightning

## Activate environment
1. Install [pipenv](https://pipenv.pypa.io/en/latest/install/)
2. Install python packages

```pipenv install```

## Model training

```python training.py data.num_workers=10 data.batch_size=32```

For more parameters, check [config.yml](config.yml).

## Model evaluation

```python evaluation.py data.num_workers=10 data.batch_size=32 logging.best_model_path=outputs/2020-10-31/17-15-03/best_model/```
