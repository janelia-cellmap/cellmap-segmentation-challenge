"""Unit tests for train() covering debug_memory, checkpoint saving, and visualization."""
import contextlib
import os
import types
from unittest import mock

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from cellmap_data.utils import get_fig_dict as _real_get_fig_dict

from cellmap_segmentation_challenge.utils.security import Config

# Import train once at module load time so torchvision's one-time meta-kernel
# registration happens before any sys.modules patching in the tests.
from cellmap_segmentation_challenge.train import train as _train_fn


class _TinyModel(nn.Module):
    """Minimal 2-class 2-D conv model – forward/backward complete in microseconds."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1)

    def forward(self, x):
        return self.conv(x)


def _make_config(tmp_path, **overrides):
    """Build a minimal Config consumable by train()."""
    model = _TinyModel()
    opts = dict(
        debug_memory=False,
        base_experiment_path=str(tmp_path),
        model_save_path=os.path.join(str(tmp_path), "ckpt_{model_name}_{epoch}.pth"),
        logs_save_path=os.path.join(str(tmp_path), "logs_{model_name}"),
        datasplit_path=os.path.join(str(tmp_path), "datasplit.csv"),
        epochs=1,
        iterations_per_epoch=2,
        batch_size=1,
        learning_rate=1e-4,
        classes=["mito", "er"],
        model_name="test_model",
        model_to_load="test_model",
        model=model,
        model_kwargs={},
        load_model="latest",
        device="cpu",
        use_s3=False,
        spatial_transforms={},
        validation_prob=0.0,
        validation_time_limit=None,
        validation_batch_limit=None,
        weight_loss=False,
        use_mutual_exclusion=False,
        weighted_sampler=False,
        max_grad_norm=None,
        force_all_classes=False,
        gradient_accumulation_steps=1,
        random_seed=42,
        filter_by_scale=False,
        scheduler=None,
        criterion=torch.nn.BCEWithLogitsLoss,
        criterion_kwargs={},
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-4),
        # 2-D, no singleton dim → singleton_dim=None, spatial_dims=2
        input_array_info={"shape": (4, 4), "scale": (8, 8)},
        target_array_info={"shape": (4, 4), "scale": (8, 8)},
    )
    opts.update(overrides)
    return Config(**opts)


# ---------------------------------------------------------------------------
# Mock dataloaders
# ---------------------------------------------------------------------------

class _MockInnerLoader:
    """Infinite source of tiny identical batches."""

    def __iter__(self):
        while True:
            yield {
                "input": torch.zeros(1, 1, 4, 4),
                "target": torch.zeros(1, 2, 4, 4),
            }

    def __len__(self):
        return 1


class _MockEmptyLoader:
    """Empty val loader – skips the validation loop."""

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0


class _MockDataset:
    input_arrays = {"input": None}
    target_arrays = {"target": None}
    class_weights = {"mito": 1.0, "er": 1.0}


class _MockTrainLoader:
    def __init__(self):
        self.loader = _MockInnerLoader()
        self.dataset = _MockDataset()

    def refresh(self):
        pass


class _MockValLoader:
    def __init__(self):
        self.loader = _MockEmptyLoader()
        self.dataset = _MockDataset()

    def refresh(self):
        pass


# ---------------------------------------------------------------------------
# Shared patch builders
# ---------------------------------------------------------------------------

def _core_patches(cfg):
    """Patches for infrastructure components unrelated to the feature under test."""
    return [
        mock.patch(
            "cellmap_segmentation_challenge.train.load_safe_config",
            return_value=cfg,
        ),
        mock.patch(
            "cellmap_segmentation_challenge.train.get_dataloader",
            return_value=(_MockTrainLoader(), _MockValLoader()),
        ),
        mock.patch(
            "cellmap_segmentation_challenge.train.get_model",
            return_value=0,
        ),
        mock.patch(
            "cellmap_segmentation_challenge.train.SummaryWriter",
            return_value=mock.MagicMock(),
        ),
    ]


def _run_train(cfg):
    """Run train() with debug_memory-focused patches.

    torch.save and get_fig_dict are mocked out because they are tested
    separately in TestCheckpointSaving and TestVisualization.
    MEMORY_LOG_STEPS=1 makes the in-loop debug branch fire on epoch_iter=1
    without needing 100 iterations.
    """
    patches = _core_patches(cfg) + [
        mock.patch("torch.save"),
        mock.patch(
            "cellmap_segmentation_challenge.train.get_fig_dict",
            return_value={},
        ),
        mock.patch.dict(os.environ, {"MEMORY_LOG_STEPS": "1"}),
    ]

    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        # datasplit.csv must exist so train() skips the CSV-generation step
        open(cfg.datasplit_path, "w").close()
        _train_fn("dummy_config.py")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDebugMemoryFlag:
    def test_debug_memory_true_with_objgraph(self, tmp_path):
        """debug_memory=True with objgraph available completes without error."""
        mock_objgraph = types.ModuleType("objgraph")
        mock_objgraph.show_growth = mock.Mock()

        cfg = _make_config(tmp_path, debug_memory=True)

        with mock.patch.dict("sys.modules", {"objgraph": mock_objgraph}):
            _run_train(cfg)

        # Pre-loop baseline call + one in-loop call at epoch_iter=1
        # (MEMORY_LOG_STEPS=1 means every iteration after the first triggers it)
        assert mock_objgraph.show_growth.call_count == 2

    def test_debug_memory_true_without_objgraph(self, tmp_path):
        """debug_memory=True falls back gracefully when objgraph is missing."""
        cfg = _make_config(tmp_path, debug_memory=True)

        # sys.modules[key]=None makes `import key` raise ImportError
        with mock.patch.dict("sys.modules", {"objgraph": None}):
            _run_train(cfg)  # must not raise

    def test_debug_memory_false_never_calls_objgraph(self, tmp_path):
        """With debug_memory=False (default), objgraph.show_growth is never called."""
        mock_objgraph = types.ModuleType("objgraph")
        mock_objgraph.show_growth = mock.Mock()

        cfg = _make_config(tmp_path, debug_memory=False)

        with mock.patch.dict("sys.modules", {"objgraph": mock_objgraph}):
            _run_train(cfg)

        mock_objgraph.show_growth.assert_not_called()


class TestCheckpointSaving:
    def test_checkpoint_file_created(self, tmp_path):
        """train() writes a loadable checkpoint at the formatted path after each epoch."""
        cfg = _make_config(tmp_path)

        # torch.save is NOT mocked – we want the real file to appear on disk.
        patches = _core_patches(cfg) + [
            mock.patch(
                "cellmap_segmentation_challenge.train.get_fig_dict",
                return_value={},
            ),
        ]

        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            open(cfg.datasplit_path, "w").close()
            _train_fn("dummy_config.py")

        ckpt = tmp_path / "ckpt_test_model_1.pth"
        assert ckpt.exists(), "checkpoint should be written after epoch 1"

        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        fresh_model = _TinyModel()
        fresh_model.load_state_dict(state)  # raises if keys or shapes are wrong
        assert set(state.keys()) == set(cfg.model.state_dict().keys())


class TestVisualization:
    def test_get_fig_dict_called_with_last_training_batch(self, tmp_path):
        """train() passes the last training batch to get_fig_dict (no validation)."""
        cfg = _make_config(tmp_path)

        # Use wraps= so the real get_fig_dict runs (producing actual figures)
        # while we can still inspect what arguments it received.
        with mock.patch(
            "cellmap_segmentation_challenge.train.get_fig_dict",
            wraps=_real_get_fig_dict,
        ) as spy:
            patches = _core_patches(cfg) + [mock.patch("torch.save")]
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                open(cfg.datasplit_path, "w").close()
                _train_fn("dummy_config.py")

        spy.assert_called_once()
        inputs_arg, targets_arg, outputs_arg, classes_arg = spy.call_args.args
        # inputs: (batch=1, channels=1, h=4, w=4) – single raw EM channel
        assert inputs_arg.shape == (1, 1, 4, 4)
        # targets / outputs: (batch=1, num_classes=2, h=4, w=4)
        assert targets_arg.shape == (1, 2, 4, 4)
        assert outputs_arg.shape == (1, 2, 4, 4)
        assert classes_arg == cfg.classes

    def test_figures_closed_after_epoch(self, tmp_path):
        """train() closes every matplotlib figure it creates, leaving no open figures."""
        cfg = _make_config(tmp_path)

        open_before = set(plt.get_fignums())

        patches = _core_patches(cfg) + [
            mock.patch("torch.save"),
            mock.patch(
                "cellmap_segmentation_challenge.train.get_fig_dict",
                wraps=_real_get_fig_dict,
            ),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            open(cfg.datasplit_path, "w").close()
            _train_fn("dummy_config.py")

        open_after = set(plt.get_fignums())
        leaked = open_after - open_before
        assert not leaked, f"figures were not closed after training: {leaked}"
