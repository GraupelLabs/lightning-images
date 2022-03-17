# Lightning Images
Training scripts for an end-to-end image classification based on Pytorch-Lightning with support of training in the cloud powered by [Grid AI](https://grid.ai).

## Activate environment
1. Install [pipenv](https://pipenv.pypa.io/en/latest/install/)
2. Install python packages

    ```
    pipenv install
    ```

3. Freeze pip dependencies (for cloud training only)
    ```
    pipenv lock -r > requirements.txt
    ```

## Model training

### Locally

First, create a config.yaml file by copying the template file:

```bash
cp config.template.yaml config.yaml
```

To start training locally, execute the [training.py](training.py) script and pass configuration parameters to it. For example,

```bash
python training.py \
    data.num_workers=10 \
    data.batch_size=128 \
    data.dataset_path=/path/to/data
```

For more parameters, check [config.yml](config.yaml).

### Cloud

Lightning Images is tested to work with [Grid AI](https://grid.ai) for cloud training. Similar to running locally, create config.yaml before executing the script.

```bash
grid run --name --localdir \
    --instance_type 2_M60_8GB \
    --datastore_name cifar5 \
    --datastore_version 1 \
    --framework lightning \
    --gpus 2 \
    training.py \
    data.num_workers=10 \
    data.batch_size=128 \
    data.dataset_path=/datastores/cifar5 \
    trainer.gpus=2
```

Note: your dataset has to be created prior to starting training. For example:

```bash
grid datastore create /path/to/data --name cifar5
```


## Model evaluation

```bash
python evaluation.py data.num_workers=10 data.batch_size=32 logging.best_model_path=outputs/2022-03-16/17-15-03/best_model/
```
