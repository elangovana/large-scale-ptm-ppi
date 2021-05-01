from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import torch

from inference.ensemble_predictor import EnsemblePredictor


class TestEnsemblePredictor(TestCase):

    def test_predict(self):
        """
        Simple base case with single ensemble
        """
        # Arrange
        # output mock data
        predictions = torch.tensor([1, 0])
        # return a batch of results
        confidence_scores = torch.tensor([[0.0715140, 0.08877321],
                                          [0.0815140, 0.07877321]])
        # mock predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = predictions, confidence_scores

        mock_model = MagicMock
        models = mock_model

        sut = EnsemblePredictor(model_wrapper=mock_predictor)

        # Act
        actual_predictions, actual_confidence = sut.predict(models, input)

        # Assert
        np.testing.assert_array_equal(confidence_scores.numpy(), actual_confidence.numpy())
        np.testing.assert_array_equal(predictions.numpy(), actual_predictions.numpy())



    def test_predict_2_different_confidence(self):
        """
        Simple base case with 2 items in ensemble
        """
        # Arrange
        # output mock data
        predictions = torch.tensor([1, 0])
        confidence_scores_1 = torch.tensor([[0.0, 1], [1.0, 0.0]])
        confidence_scores_2 = torch.tensor([[0.05, .95], [.80, 0.2]])

        expected_confidence_scores = torch.tensor([[0.05 / 2, 1.95 / 2], [1.8 / 2, 0.2 / 2]])

        mock_model_1 = MagicMock()
        mock_model_2 = MagicMock()

        models = [mock_model_1, mock_model_2]

        # mock predictor
        mock_model_wrapper = MagicMock()

        def mock_model_wrapper_call(m, d, h):
            return (predictions, confidence_scores_1) if m == mock_model_1 else (predictions, confidence_scores_2)

        mock_model_wrapper.predict.side_effect = mock_model_wrapper_call

        # 2 models
        sut = EnsemblePredictor(model_wrapper=mock_model_wrapper)

        # Act
        actual_predictions, actual_confidence = sut.predict(models, input)

        # Assert
        np.testing.assert_array_equal(expected_confidence_scores.numpy(), actual_confidence.numpy())
        np.testing.assert_array_equal(predictions.numpy(), actual_predictions.numpy())
