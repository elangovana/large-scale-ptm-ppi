import json
import os
import tempfile
from unittest import TestCase

from dataset_builder import DatasetBuilder
from train_builder import TrainBuilder


class TestSitTrain(TestCase):

    def test_run_with_no_exception_aimed(self):
        # Arrange
        train_data_file = os.path.join(os.path.dirname(__file__), "sample_data", "train_data", "aimed.json")
        tempdir = tempfile.mkdtemp()
        batch = 3

        # Bert Config
        vocab_size = 20000
        sequence_len = 20

        bert_config = {"vocab_size": vocab_size, "hidden_size": 10, "num_hidden_layers": 1,
                       "tokenisor_max_seq_len": sequence_len,
                       "num_attention_heads": 1}

        # Additional args
        additional_args = {"model_config": json.dumps(bert_config)
            , "tokenisor_data_dir": os.path.join(os.path.dirname(__file__), "sample_data", "tokensior_data")
                           }

        dataset_factory = "datasets.aimed_dataset_factory.AimedDatasetFactory"
        model_factory = "models.bert_model_factory.BertModelFactory"

        self._run_train(train_data_file, dataset_factory, model_factory, batch, tempdir, additional_args)


    def test_run_with_no_exception_ppi_multiclass(self):
        # Arrange
        train_data_file = os.path.join(os.path.dirname(__file__), "sample_data", "train_data_ppi_multiclass")
        tempdir = tempfile.mkdtemp()
        batch = 3

        # Bert Config
        vocab_size = 20000
        sequence_len = 20
        bert_config = {"vocab_size": vocab_size, "hidden_size": 10, "num_hidden_layers": 1,
                       "tokenisor_max_seq_len": sequence_len,
                       "num_attention_heads": 1}

        # Additional args
        additional_args = {"model_config": json.dumps(bert_config)
            , "tokenisor_data_dir": os.path.join(os.path.dirname(__file__), "sample_data", "tokensior_data")
                           }

        dataset_factory = "datasets.ppi_multiclass_dataset_factory.PpiMulticlassDatasetFactory"
        model_factory = "models.bert_model_factory.BertModelFactory"

        self._run_train(train_data_file, dataset_factory, model_factory, batch, tempdir, additional_args)

    def _run_train(self, train_data_file, dataset_factory, model_factory, batch, artifacts_dir,
                   additional_args):
        # Dataset builder
        dataset_builder = DatasetBuilder(val_data=train_data_file,
                                         dataset_factory_name=dataset_factory,
                                         tokenisor_factory_name=model_factory,
                                         train_data=train_data_file,
                                         batch_size=batch, num_workers=1,
                                         addition_args_dict=additional_args)
        # Train builder
        train_builder = TrainBuilder(model_factory_name=model_factory,
                                     scorers=dataset_builder.get_scorers(),
                                     num_classes=dataset_builder.num_classes(),
                                     checkpoint_dir=artifacts_dir, epochs=2, grad_accumulation_steps=1,
                                     early_stopping_patience=2, model_dir=artifacts_dir,
                                     addition_args_dict=additional_args
                                     )
        trainer = train_builder.get_trainer()
        # Run training
        trainer.run_train(train_iter=dataset_builder.get_train_dataloader(),
                          validation_iter=dataset_builder.get_val_dataloader(),
                          model_network=train_builder.get_network(),
                          loss_function=train_builder.get_loss_function(),
                          optimizer=train_builder.get_optimiser(),
                          pos_label=dataset_builder.positive_label_index()
                          )
