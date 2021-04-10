# Protien protein interaction AIMed
AIMed protein protein relation extraction


## Download AIMed dataset

1. Download from ftp://ftp.cs.utexas.edu/pub/mooney/bio-data/interactions.tar.gz

2. Convert the raw dataset into XML for using instructions in http://mars.cs.utu.fi/PPICorpora/
      ```bash
      convert_aimed.py -i  aimed_interactions_input_dir -o aimed.xml

      ```

## Run

### Step 1: Convert xml AIMed to flattened json

```bash
python src/preprocessors/aimed_json_converter.py --inputfile tests/sample_data/aimed.xml --outputfile aimed.json
```

### Step 2: 10 fold split

You can either choose random split, or split by unique documents

[Option R - Random ] This randomly splits into n folds

   ```bash
   python src/preprocessors/kfold_aimed_json_splitter.py --inputfile aimed.json --outputdir temp_data/kfolds_random  --kfoldLabelColumn interacts --k 10
   ```

[Option U - Unique Document] This splits into n folds, taking into account document id uniqueness

   ```bash
   python src/preprocessors/kfold_aimed_json_splitter.py --inputfile aimed.json --outputdir temp_data/kfolds_unique  --kfoldLabelColumn interacts --k 10  --kfoldDocId documentId
   ```