[![Build Status](https://travis-ci.org/elangovana/ppi-aimed.svg?branch=main)](https://travis-ci.org/elangovana/ppi-aimed)

# Large-scale protein-protein post-translational modification extraction with distant supervision and confidence calibrated BioBERT

![docs/images/Overview.png](docs/images/Overview.png)

## PTM-PPI Dataset relation extraction

- For data preparation, see https://github.com/elangovana/PPI-typed-relation-extractor
- For training see, [notebooks/ppi_multiclass_sagemaker_bert.ipynb](notebooks/ppi_multiclass_sagemaker_bert.ipynb)
- For large scale prediction,
  see [notebooks/ppi_multiclass_large_scale_prediction.ipynb](notebooks/ppi_multiclass_large_scale_prediction.ipynb)

## AIMed PPI relation extraction

### Download AIMed dataset

1. Download from ftp://ftp.cs.utexas.edu/pub/mooney/bio-data/interactions.tar.gz

2. Convert the raw dataset into XML for using instructions in http://mars.cs.utu.fi/PPICorpora/
      ```bash
      convert_aimed.py -i  aimed_interactions_input_dir -o aimed.xml

      ```

### Run

#### Step 1: Convert xml AIMed to flattened json

```bash
python src/preprocessors/aimed_json_converter.py --inputfile tests/sample_data/aimed.xml --outputfile aimed.json
```

#### Step 2: 10 fold split

You can either choose random split, or split by unique documents

[Option R - Random ] This randomly splits into n folds

   ```bash
   python src/preprocessors/kfold_aimed_json_splitter.py --inputfile aimed.json --outputdir temp_data/kfolds_random  --kfoldLabelColumn interacts --k 10
   ```

[Option U - Unique Document] This splits into n folds, taking into account document id uniqueness

   ```bash
   python src/preprocessors/kfold_aimed_json_splitter.py --inputfile aimed.json --outputdir temp_data/kfolds_unique  --kfoldLabelColumn interacts --k 10  --kfoldDocId documentId
   ```

#### Step 3: Run training

```bash
python src/main_train.py --datasetfactory datasets.aimed_dataset_factory.AimedDatasetFactory --traindir temp_data/kfold_unique --modeldir temp_data --outdir temp_data --kfoldtrainprefix train  --model_config '{"vocab_size": 20000, "hidden_size": 10, "num_hidden_layers": 1, "num_attention_heads": 1, "num_labels": 2}' --tokenisor_data_dir tests/sample_data/tokensior_data --epochs 1 --numworkers 1
```