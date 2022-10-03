from pathlib import Path
import torch
import requests
from tqdm import tqdm

from allennlp.common.testing import ModelTestCase
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import Token
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField
from allennlp.data import Batch

from gector.gec_model import GecBERTModel
from gector.bert_token_embedder import PretrainedBertEmbedder
from gector.tokenizer_indexer import PretrainedBertIndexer


class TestGecModel(ModelTestCase):
    """Test class for GecModel"""

    def setup_method(self):
        super().setup_method()
        self.vocab_path = "data/output_vocabulary"
        self.vocab = Vocabulary.from_files(self.vocab_path)
        self.model_name = "roberta-base"
        model_url = "https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th"
        test_fixtures_dir_path = Path(__file__).parent.parent / "test_fixtures"
        model_path = test_fixtures_dir_path / "roberta_1_gectorv2.th"
        if not model_path.exists():
            response = requests.get(model_url)
            with model_path.open("wb") as out_fp:
                # Write out data with progress bar
                for data in tqdm(response.iter_content()):
                    out_fp.write(data)
        assert model_path.exists()
        self.model_path = model_path
        sentence1 = "I run to a stores every day."
        sentence2 = "the quick brown foxes jumps over a elmo's laziest friend"
        # This micmics how batches of requests are constructed in predict.py's predict_for_file function
        self.input_data = [sentence1, sentence2]
        self.input_data = [sentence.split() for sentence in self.input_data]

    def test_gec_model_prediction(self):
        """Test simple prediction with GecBERTModel"""
        gec_model = GecBERTModel(
            vocab_path=self.vocab_path,
            model_paths=[self.model_path],
            max_len=50,
            iterations=5,
            min_error_probability=0.0,
            lowercase_tokens=0,
            model_name="roberta",
            special_tokens_fix=1,
            log=False,
            confidence=0,
            del_confidence=0,
            is_ensemble=0,
            weigths=None,
        )
        final_batch, total_updates = gec_model.handle_batch(self.input_data)
        # subject verb agreement is not fixed in the second sentence when predicting using GecModel
        # (i.e.) brown foxes jump
        assert final_batch == [
            ["I", "run", "to", "the", "stores", "every", "day."],
            [
                "The",
                "quick",
                "brown",
                "foxes",
                "jumps",
                "over",
                "Elmo's",
                "laziest",
                "friend",
            ],
        ]
        assert total_updates == 2
