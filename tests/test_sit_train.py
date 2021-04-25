import json
import os
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock

from builder import Builder


class TestSitTrain(TestCase):

    def test_run_with_no_exception_aimed(self):
        # Arrange
        train_data_file = os.path.join(os.path.dirname(__file__), "sample_data", "train_data", "aimed.json")
        tempdir = tempfile.mkdtemp()
        batch = 3

        # Bert Config
        vocab_size = 20000
        sequence_len = 20
        num_classes = 2
        bert_config = {"vocab_size": vocab_size, "hidden_size": 10, "num_hidden_layers": 1,
                       "num_attention_heads": 1, "num_labels": num_classes}

        # Mock tokenisor
        mock_tokenisor = MagicMock()
        mock_tokenisor.tokenize.side_effect = lambda x: x.split(" ")
        mock_tokenisor.convert_tokens_to_ids = lambda x: [i for i, _ in enumerate(x)]

        # Additional args
        additional_args = {"model_config": json.dumps(bert_config)
            , "tokenisor_data_dir": os.path.join(os.path.dirname(__file__), "sample_data", "tokensior_data")
                           }

        # Builder
        b = Builder(train_data=train_data_file, val_data=train_data_file,
                    dataset_factory_name="datasets.aimed_dataset_factory.AimedDatasetFactory",
                    model_factory_name="models.bert_model_factory.BertModelFactory",
                    checkpoint_dir=tempdir, epochs=2, grad_accumulation_steps=1, num_workers=1,
                    early_stopping_patience=2, batch_size=batch,
                    max_seq_len=sequence_len, model_dir=tempdir, addition_args_dict=additional_args)

        trainer = b.get_trainer()

        # Get data loaders
        train_dataloader, val_dataloader = b.get_train_val_dataloader()

        # Act
        # Run training
        trainer.run_train(train_iter=train_dataloader,
                          validation_iter=val_dataloader,
                          model_network=b.get_network(),
                          loss_function=b.get_loss_function(),
                          optimizer=b.get_optimiser(), pos_label=b.get_pos_label_index())

    def test_run_with_no_exception_ppi_multiclass(self):
        # Arrange
        train_data_file = os.path.join(os.path.dirname(__file__), "sample_data", "train_data_ppi_multiclass")
        tempdir = tempfile.mkdtemp()
        batch = 3

        # Bert Config
        vocab_size = 20000
        sequence_len = 20
        num_classes = 7
        bert_config = {"vocab_size": vocab_size, "hidden_size": 10, "num_hidden_layers": 1,
                       "num_attention_heads": 1, "num_labels": num_classes}

        # Mock tokenisor
        mock_tokenisor = MagicMock()
        mock_tokenisor.tokenize.side_effect = lambda x: x.split(" ")
        mock_tokenisor.convert_tokens_to_ids = lambda x: [i for i, _ in enumerate(x)]

        # Additional args
        additional_args = {"model_config": json.dumps(bert_config)
            , "tokenisor_data_dir": os.path.join(os.path.dirname(__file__), "sample_data", "tokensior_data")
                           }

        # Builder
        b = Builder(train_data=train_data_file, val_data=train_data_file,
                    dataset_factory_name="datasets.ppi_multiclass_dataset_factory.PpiMulticlassDatasetFactory",
                    model_factory_name="models.bert_model_factory.BertModelFactory",
                    checkpoint_dir=tempdir, epochs=2, grad_accumulation_steps=1, num_workers=1,
                    early_stopping_patience=2, batch_size=batch,
                    max_seq_len=sequence_len, model_dir=tempdir, addition_args_dict=additional_args)

        trainer = b.get_trainer()

        # Get data loaders
        train_dataloader, val_dataloader = b.get_train_val_dataloader()

        # Act
        # Run training
        trainer.run_train(train_iter=train_dataloader,
                          validation_iter=val_dataloader,
                          model_network=b.get_network(),
                          loss_function=b.get_loss_function(),
                          optimizer=b.get_optimiser(), pos_label=b.get_pos_label_index())
