"""Test cho Task D3: Nạp và lọc GloVe theo từ vựng."""
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.data.vocab import Vocab
from src.models.glove import build_glove_matrix


def test_build_glove_matrix_with_mock_glove():
    vocab = Vocab({"<pad>": 0, "<unk>": 1, "great": 2, "movie": 3, "oovword": 4})
    embed_dim = 4

    with tempfile.TemporaryDirectory() as tmp_dir:
        glove_txt = Path(tmp_dir) / "mock_glove.txt"
        content = "great 0.1 0.2 0.3 0.4\nmovie 0.5 0.6 0.7 0.8\nother 0.9 1.0 1.1 1.2\n"
        glove_txt.write_text(content, encoding="utf-8")

        npy_path = Path(tmp_dir) / "glove_vocab.npy"
        matrix, coverage = build_glove_matrix(
            vocab=vocab,
            glove_path=glove_txt,
            embed_dim=embed_dim,
            output_path=npy_path,
            min_coverage=0.0,
        )

        assert matrix.shape == (5, embed_dim)
        assert np.allclose(matrix[0], np.zeros(embed_dim))
        assert np.allclose(matrix[2], [0.1, 0.2, 0.3, 0.4])
        assert np.allclose(matrix[3], [0.5, 0.6, 0.7, 0.8])
        assert coverage == 0.5
        assert npy_path.exists()
        loaded = np.load(npy_path)
        assert np.allclose(loaded, matrix)

        # Kiểm tra ngưỡng min_coverage mặc định (0.80) báo lỗi khi coverage quá thấp
        with pytest.raises(ValueError, match="Coverage GloVe"):
            build_glove_matrix(vocab=vocab, glove_path=glove_txt, embed_dim=embed_dim, min_coverage=0.80)


def test_build_glove_matrix_allow_random_when_missing():
    vocab = Vocab({"<pad>": 0, "<unk>": 1, "film": 2})
    with tempfile.TemporaryDirectory() as tmp_dir:
        missing_path = Path(tmp_dir) / "missing_glove.txt"
        with pytest.raises(FileNotFoundError):
            build_glove_matrix(vocab=vocab, glove_path=missing_path, embed_dim=8, allow_random=False)

        matrix, cov = build_glove_matrix(vocab=vocab, glove_path=missing_path, embed_dim=8, allow_random=True)
        assert matrix.shape == (3, 8)
        assert np.allclose(matrix[0], np.zeros(8))
        assert cov == 0.0
