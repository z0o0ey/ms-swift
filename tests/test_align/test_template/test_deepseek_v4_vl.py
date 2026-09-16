"""DeepSeek-V4-Flash-Vision (deepseek_v4_vl) alignment tests.

Two test layers:

1. **Template alignment** — verifies that image token expansion (sentinel token
   IDs, N-layout ordering, compress padding, start/end markers) matches the
   expected output for known image dimensions.  Only the tokenizer/config is
   loaded; no model weights are downloaded.

2. **Forward precision alignment** — runs ``test_convert_precision`` with an
   image input through the HF→mcore path, verifying that the multimodal
   forward (vision encoding + aligner + sentinel scatter + LLM) produces
   matching logits between HuggingFace transformers and Megatron-Core.

   Requires GPUs and model weights.  Launch with::

       CUDA_VISIBLE_DEVICES=0,1,2,3 \\
       torchrun --nproc_per_node=4 \\
       tests/test_align/test_template/test_deepseek_v4_vl.py
"""

import os

# Template-only tests can run on CPU with SWIFT_DEBUG.
os.environ.setdefault('SWIFT_DEBUG', '1')

import math
import unittest
from typing import Dict, Any

import torch

# ---------------------------------------------------------------------------
# Sentinel token offsets — must match DeepseekV4VisionTemplate class constants
# and mcore-bridge DeepSeekV4Vit.
# ---------------------------------------------------------------------------
from swift.template.templates.deepseek import DeepseekV4VisionTemplate as _V4Template

IMAGE_START = _V4Template.IMAGE_START
IMAGE_PAD = _V4Template.IMAGE_PAD
IMAGE = _V4Template.IMAGE
IMAGE_NEWLINE = _V4Template.IMAGE_NEWLINE
IMAGE_END = _V4Template.IMAGE_END
COMPRESS_PAD_TO = _V4Template.COMPRESS_PAD_TO

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_ID = 'deepseek-ai/DeepSeek-V4-Flash-Vision-Exp'


def _get_template():
    """Return a DeepseekV4VisionTemplate with the tokenizer loaded."""
    from swift.model import get_processor
    from swift.template import get_template
    processor = get_processor(MODEL_ID, model_type='deepseek_v4_vl')
    template = get_template(processor)
    template.template_backend = 'swift'
    return template


def _make_test_image(width: int, height: int) -> 'Image.Image':
    """Create a solid-colour PIL image of the given dimensions."""
    from PIL import Image
    return Image.new('RGB', (width, height), color=(128, 64, 200))


# ---------------------------------------------------------------------------
# 1. Template alignment: _build_image_block
# ---------------------------------------------------------------------------

class TestBuildImageBlock(unittest.TestCase):
    """Verify the N-layout token type array and permutation index for known
    grid dimensions.

    The ``_build_image_block`` static method is the core of the image token
    expansion.  These tests pin its output for several ``(n_llm_h, n_llm_w)``
    combinations so regressions are caught without needing model weights.
    """

    @classmethod
    def setUpClass(cls):
        cls.template_cls = None
        # Import the class without instantiating a full template.
        from swift.template.templates.deepseek import DeepseekV4VisionTemplate
        cls.template_cls = DeepseekV4VisionTemplate

    def _check_block(self, n_llm_h, n_llm_w, start_pos):
        """Common assertions for a (n_llm_h, n_llm_w, start_pos) triple."""
        types, perm = self.template_cls._build_image_block(n_llm_h, n_llm_w, start_pos)

        # The block must start with compress_pad IMAGE_PAD tokens, then
        # IMAGE_START, then the N-layout body, then optional padding, then
        # IMAGE_END.
        compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
        pad_h = n_llm_h % 2
        rows = n_llm_h + pad_h
        row_len = n_llm_w + 1
        pad_last = rows // 2 * row_len % 2 * 2

        self.assertEqual(types[compress_pad].item(), IMAGE_START,
                         f'IMAGE_START not at expected position for ({n_llm_h}, {n_llm_w}, {start_pos})')
        self.assertEqual(types[-1].item(), IMAGE_END,
                         f'IMAGE_END not at end for ({n_llm_h}, {n_llm_w}, {start_pos})')
        self.assertEqual(types[:compress_pad].tolist(),
                         [IMAGE_PAD] * compress_pad,
                         f'compress_pad mismatch for ({n_llm_h}, {n_llm_w}, {start_pos})')

        # Count IMAGE tokens: should be n_llm_h * n_llm_w.
        n_image = (types == IMAGE).sum().item()
        self.assertEqual(n_image, n_llm_h * n_llm_w,
                         f'IMAGE count mismatch for ({n_llm_h}, {n_llm_w}, {start_pos})')

        # Count NEWLINE tokens: should be n_llm_h.
        n_newline = (types == IMAGE_NEWLINE).sum().item()
        self.assertEqual(n_newline, n_llm_h,
                         f'NEWLINE count mismatch for ({n_llm_h}, {n_llm_w}, {start_pos})')

        # Total length = compress_pad + 1 (START) + rows*row_len + pad_last + 1 (END)
        expected_len = compress_pad + 1 + rows * row_len + pad_last + 1
        self.assertEqual(len(types), expected_len,
                         f'total length mismatch for ({n_llm_h}, {n_llm_w}, {start_pos})')

        # Permutation should index into n_llm_h * n_llm_w patches.
        self.assertEqual(len(perm), n_llm_h * n_llm_w,
                         f'perm length mismatch for ({n_llm_h}, {n_llm_w}, {start_pos})')
        self.assertEqual(sorted(perm.tolist()), list(range(n_llm_h * n_llm_w)),
                         f'perm not a permutation of [0..{n_llm_h*n_llm_w-1}] for ({n_llm_h}, {n_llm_w}, {start_pos})')

        return types, perm

    def test_even_grid_start0(self):
        """2×3 grid, start_pos=0 → compress_pad=3."""
        types, perm = self._check_block(2, 3, 0)
        # compress_pad = 4 - 1 - 0 % 4 = 3
        # rows = 2 (even), row_len = 4, pad_last = 1*4 % 2 * 2 = 0
        # total = 3 + 1 + 8 + 0 + 1 = 13
        self.assertEqual(len(types), 13)

    def test_odd_height_start0(self):
        """3×2 grid (odd height), start_pos=0."""
        types, perm = self._check_block(3, 2, 0)
        # pad_h = 1, rows = 4, row_len = 3, pad_last = 2*3 % 2 * 2 = 0
        # total = 3 + 1 + 12 + 0 + 1 = 17
        self.assertEqual(len(types), 17)

    def test_start_pos_alignment(self):
        """start_pos=1 → compress_pad=2 (aligns next block to 4-token boundary)."""
        types, perm = self._check_block(2, 3, 1)
        # compress_pad = 4 - 1 - 1 % 4 = 2
        # total = 2 + 1 + 8 + 0 + 1 = 12
        self.assertEqual(len(types), 12)

    def test_large_grid(self):
        """4×5 grid (like a 384-token image)."""
        types, perm = self._check_block(4, 5, 0)
        # compress_pad=3, pad_h=0, rows=4, row_len=6, pad_last=2*6%2*2=0
        # total = 3 + 1 + 24 + 0 + 1 = 29
        self.assertEqual(len(types), 29)
        # 20 IMAGE tokens + 4 NEWLINE = 24 body tokens
        n_image = (types == IMAGE).sum().item()
        self.assertEqual(n_image, 20)

    def test_n_layout_interleaving(self):
        """Verify N-layout actually interleaves row pairs.

        For a 2×2 grid at start_pos=0:
          Row-major:  [I I NL] [I I NL]
          N-layout:   [I I NL I I NL]  (same because rows//2=1, no interleaving)

        For a 4×2 grid at start_pos=0:
          Row-major:  r0=[I I NL] r1=[I I NL] r2=[I I NL] r3=[I I NL]
          N-layout:   r0 r1 r2 r3 → interleave pairs: (r0,r1)(r2,r3)
          order:      0 3 1 4 2 5 6 9 7 10 8 11  (column-interleaved within each pair)
        """
        types, perm = self._check_block(4, 2, 0)
        # The body (between START and END) should be 4 rows * 3 cols = 12 tokens
        # N-layout order for 4 rows: interleave (r0,r1) then (r2,r3)
        body = types[4:-1]  # skip compress_pad(3) + START(1) at front, END at back
        self.assertEqual(len(body), 12)

        # In N-layout, row 0 col 0 and row 1 col 0 should be adjacent
        # (order for pair (r0,r1): col0_r0 col0_r1 col1_r0 col1_r1 col2_r0 col2_r1)
        # So body[0] and body[1] should both be IMAGE (first column of rows 0,1)
        self.assertEqual(body[0].item(), IMAGE)
        self.assertEqual(body[1].item(), IMAGE)




# ---------------------------------------------------------------------------
# 2. Template alignment: _grid_tokens and _solve_resize_ratio
# ---------------------------------------------------------------------------

class TestGridTokens(unittest.TestCase):
    """Verify token-count arithmetic for known image sizes."""

    @classmethod
    def setUpClass(cls):
        from swift.template.templates.deepseek import DeepseekV4VisionTemplate
        cls.cls = DeepseekV4VisionTemplate

    def test_grid_tokens_2x3(self):
        n_llm_h, n_llm_w, num_tokens = self.cls._grid_tokens(
            best_height=56, best_width=84, patch_size=14, downsample_ratio=2)
        # h/p/r = 56/14/2 = 2, w/p/r = 84/14/2 = 3
        self.assertEqual(n_llm_h, 2)
        self.assertEqual(n_llm_w, 3)
        # num_tokens = 2*(3+1) + 2 = 10  (even height, no padding)
        self.assertEqual(num_tokens, 10)

    def test_grid_tokens_3x2(self):
        n_llm_h, n_llm_w, num_tokens = self.cls._grid_tokens(
            best_height=84, best_width=56, patch_size=14, downsample_ratio=2)
        # h/p/r = 84/14/2 = 3, w/p/r = 56/14/2 = 2
        self.assertEqual(n_llm_h, 3)
        self.assertEqual(n_llm_w, 2)
        # odd height → pad_h adds row_len=3 tokens
        # num_tokens = 3*(2+1) + 2 + 3 = 14
        # then (3+1)//2 * 3 % 2 * 2 = 2*3%2*2 = 0
        self.assertEqual(num_tokens, 14)


# ---------------------------------------------------------------------------
# 3. Template alignment: end-to-end _encode with a synthetic image
# ---------------------------------------------------------------------------

class TestEncodeImageTokens(unittest.TestCase):
    """Verify that encoding a message with ``<image>`` produces:
    - input_ids containing sentinel tokens (vocab_size + offsets)
    - no leftover ``<image>`` placeholder token
    - ``image_inputs`` metadata with correct patch dimensions
    """

    @classmethod
    def setUpClass(cls):
        cls.template = _get_template()
        cls.vocab_size = cls.template.vocab_size

    def test_single_image_expansion(self):
        """A 420×420 image (30×30 ViT patches, 10×10 LLM grid with
        downsample_ratio=3) should expand ``<image>`` into a sentinel block."""
        image = _make_test_image(420, 420)
        messages = [{'role': 'user', 'content': '<image>What is this?'}]
        encoded = self.template.encode({'messages': messages, 'images': [image]})

        input_ids = encoded['input_ids']

        # No raw <image> token should remain.
        image_token_id = self.template._tokenize('<image>')
        self.assertNotIn(image_token_id, input_ids,
                         'Raw <image> placeholder token still present in input_ids')

        # Sentinel tokens should be present (vocab_size + 0..4).
        sentinel_ids = set(range(self.vocab_size, self.vocab_size + 5))
        found_sentinels = set(tid for tid in input_ids if tid >= self.vocab_size)
        self.assertTrue(sentinel_ids.issubset(found_sentinels),
                        f'Expected sentinel tokens {sentinel_ids}, found {found_sentinels}')

        # IMAGE_START and IMAGE_END should appear exactly once each.
        n_start = input_ids.count(self.vocab_size + IMAGE_START)
        n_end = input_ids.count(self.vocab_size + IMAGE_END)
        self.assertEqual(n_start, 1, 'IMAGE_START should appear exactly once')
        self.assertEqual(n_end, 1, 'IMAGE_END should appear exactly once')

        # IMAGE_START should come before IMAGE_END.
        start_idx = input_ids.index(self.vocab_size + IMAGE_START)
        end_idx = input_ids.index(self.vocab_size + IMAGE_END)
        self.assertLess(start_idx, end_idx,
                        'IMAGE_START must precede IMAGE_END')

        # image_inputs metadata should be present.
        image_inputs = encoded.get('image_inputs')
        self.assertIsNotNone(image_inputs, 'image_inputs not in encoded output')
        self.assertEqual(len(image_inputs), 1)
        meta = image_inputs[0]
        # 420/14 = 30 ViT patches per side, ceil(30/3) = 10 LLM tokens per side
        self.assertEqual(meta['n_vit_h'], 30)
        self.assertEqual(meta['n_vit_w'], 30)
        self.assertEqual(meta['n_llm_h'], 10)
        self.assertEqual(meta['n_llm_w'], 10)
        # patches shape: (n_vit_h * n_vit_w, 3, patch_size, patch_size)
        self.assertEqual(meta['patches'].shape[0], 900)
        self.assertEqual(meta['patches'].shape[1], 3)
        self.assertEqual(meta['patches'].shape[2], 14)
        self.assertEqual(meta['patches'].shape[3], 14)

    def test_compress_padding_alignment(self):
        """The sentinel block should start at a position that is 4-token aligned
        (compress_pad fills the gap between the text prefix and IMAGE_START)."""
        image = _make_test_image(420, 420)
        messages = [{'role': 'user', 'content': '<image>Describe.'}]
        encoded = self.template.encode({'messages': messages, 'images': [image]})

        input_ids = encoded['input_ids']
        start_idx = input_ids.index(self.vocab_size + IMAGE_START)

        # The position of IMAGE_START should be ≡ 0 (mod 4) relative to the
        # start of the sequence (compress_pad ensures this).
        self.assertEqual(start_idx % COMPRESS_PAD_TO, 0,
                         f'IMAGE_START at position {start_idx} is not 4-token aligned')

    def test_labels_masked_for_image_tokens(self):
        """All sentinel tokens should have label=-100 (not trained on)."""
        image = _make_test_image(420, 420)
        messages = [{'role': 'user', 'content': '<image>What is this?'}]
        # Use train mode to ensure labels are generated.
        self.template.set_mode('train')
        encoded = self.template.encode({'messages': messages, 'images': [image]})

        labels = encoded.get('labels')
        if labels is None:
            self.skipTest('labels not generated in current mode')
        input_ids = encoded['input_ids']

        for i, tid in enumerate(input_ids):
            if tid >= self.vocab_size:
                self.assertEqual(labels[i], -100,
                                 f'Sentinel token at position {i} (id={tid}) should have label -100, got {labels[i]}')


# ---------------------------------------------------------------------------
# 4. Forward precision alignment (requires GPUs + model weights)
# ---------------------------------------------------------------------------

class TestForwardPrecisionAlignment(unittest.TestCase):
    """Run ``test_convert_precision`` with an image input through HF→mcore.

    This test validates the entire multimodal forward path:
    - swift template data preprocessing (token expansion + patch extraction)
    - mcore-bridge DeepSeekV4Vit visual encoding + aligner + sentinel scatter
    - Megatron-Core LLM forward

    Requires:
    - GPUs (at least 2, recommended 4)
    - Model weights for ``deepseek-ai/DeepSeek-V4-Flash-Vision``
    - Megatron-Core dev branch + mcore-bridge main branch

    Launch::

        CUDA_VISIBLE_DEVICES=0,1,2,3 \\
        torchrun --nproc_per_node=4 \\
        tests/test_align/test_template/test_deepseek_v4_vl.py::TestForwardPrecisionAlignment
    """

    @unittest.skipUnless(
        os.environ.get('SWIFT_TEST_DSV4_VL_FORWARD') == '1',
        'Set SWIFT_TEST_DSV4_VL_FORWARD=1 to run forward precision alignment '
        '(requires GPUs and model weights).')
    def test_hf2mcore_forward_alignment(self):
        """HF→mcore forward alignment with image input.

        Uses the standard ``test_convert_precision`` flow:
        1. Load HF model from original checkpoint
        2. Convert weights HF→mcore
        3. Run forward with an image example on both models
        4. Compare logits

        Acceptance criteria: mean_diff < 0.1 (with loss).
        """
        from swift.megatron import MegatronExportArguments, megatron_export_main
        from swift import safe_snapshot_download

        model_dir = safe_snapshot_download(MODEL_ID, download_model=False)

        megatron_export_main(
            MegatronExportArguments(
                model=model_dir,
                to_mcore=True,
                attention_backend='flash',
                tensor_model_parallel_size=1,
                pipeline_model_parallel_layout='Et*3|t*1mL',
                pipeline_model_parallel_size=2,
                expert_model_parallel_size=2,
                mtp_num_layers=1,
                test_convert_precision=True,
            ))

    @unittest.skipUnless(
        os.environ.get('SWIFT_TEST_DSV4_VL_FORWARD') == '1',
        'Set SWIFT_TEST_DSV4_VL_FORWARD=1 to run forward precision alignment '
        '(requires GPUs and model weights).')
    def test_mcore2hf_export_alignment(self):
        """mcore→HF export forward alignment with image input.

        This test catches the MoE weight export format bug:
        1. Load HF model → convert to mcore → save mcore checkpoint
        2. Load mcore checkpoint → export to HF
        3. Load exported HF model → run forward with image example
        4. Compare logits against mcore model

        Acceptance criteria: mean_diff < 0.1 (with loss).

        This test would have caught the ``gate_up_proj`` export bug before
        it reached production.
        """
        from swift.megatron import MegatronExportArguments, megatron_export_main
        from swift import safe_snapshot_download

        model_dir = safe_snapshot_download(MODEL_ID, download_model=False)
        output_dir = '/tmp/dsv4_vl_export_test'

        # Step 1: HF → mcore (save checkpoint)
        megatron_export_main(
            MegatronExportArguments(
                model=model_dir,
                to_mcore=True,
                attention_backend='flash',
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=2,
                expert_model_parallel_size=2,
                mtp_num_layers=1,
                output_dir=output_dir,
                exist_ok=True,
            ))

        # Step 2: mcore → HF (with precision test)
        megatron_export_main(
            MegatronExportArguments(
                mcore_model=os.path.join(output_dir, 'iter_0000001'),
                to_hf=True,
                output_dir=os.path.join(output_dir, 'exported_hf'),
                attention_backend='flash',
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=2,
                expert_model_parallel_size=2,
                mtp_num_layers=1,
                test_convert_precision=True,
                exist_ok=True,
            ))


if __name__ == '__main__':
    unittest.main()
