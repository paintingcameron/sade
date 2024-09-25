# SADE: Scene-text Autoregressive Diffusion Engine

![OCR results comparison](https://github.com/paintingcameron/sade/tree/master/resources/ocr-results.png)

This repository is the accompanying code for the paper on SADE: Scene-text Autoregressive Diffusion Engine. Below are instructions for creating the mock datasets used in this paper and for training an autoregressive diffusion model. The repository used to train all OCR models was cloned from [here](https://github.com/clovaai/deep-text-recognition-benchmark).


### Installation

**Install required packaged**
```
pip install -r requirements.txt
```

### Generating mock datasets

`datagen` submodule used to generate mock datasets.

Open `datagen/scripts/data-create.py` to set dataset configurations

This repository works with LMDB datasets. To create an lmdb directory, refer to `./scripts/create_lmdb_dataset.py`.

### SADE usage

**Create necessary components**
```
./scripts/script-create-model.sh
./scripts/script-create-tokenizer.sh
./scripts/script-create-scheduler.sh
```

**Train model**
```
./scripts/script-train-model.sh
```

**Sample from model**
```
./scripts/script-sample-model.sh
```


### Output Examples

The following are some examples of outputs from SADE on mock Venezuelan number plates 

![Mock images](https://github.com/paintingcameron/sade/tree/master/resources/mock-samples.png)


### License

