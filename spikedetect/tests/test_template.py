"""Tests for spikedetect.pipeline.template."""

import numpy as np

from spikedetect.pipeline.template import TemplateMatchResult, match_template


class TestMatchTemplate:
    def test_output_shapes(self, spike_template, default_params):
        fs = default_params.fs
        stw = default_params.spike_template_width
        n_samples = 20000
        rng = np.random.default_rng(42)

        filtered = rng.normal(0, 0.01, n_samples)
        unfiltered = rng.normal(0, 0.01, n_samples)

        # Place spikes well away from edges (need stw margin on each side)
        spike_locs = np.array([3000, 8000, 13000], dtype=np.int64)
        half = stw // 2
        for loc in spike_locs:
            filtered[loc - half:loc + half + 1] += spike_template * 10

        result = match_template(
            spike_locs=spike_locs,
            spike_template=spike_template,
            filtered_data=filtered,
            unfiltered_data=unfiltered,
            spike_template_width=stw,
            fs=fs,
        )

        assert isinstance(result, TemplateMatchResult)
        assert len(result.spike_locs) == 3
        assert len(result.dtw_distances) == 3
        assert len(result.amplitudes) == 3
        assert result.filtered_candidates.shape[1] == 3
        assert result.unfiltered_candidates.shape[1] == 3
        assert result.norm_filtered_candidates.shape[1] == 3

    def test_empty_spike_locs(self, spike_template, default_params):
        stw = default_params.spike_template_width
        fs = default_params.fs
        filtered = np.zeros(10000)
        unfiltered = np.zeros(10000)

        result = match_template(
            spike_locs=np.array([], dtype=np.int64),
            spike_template=spike_template,
            filtered_data=filtered,
            unfiltered_data=unfiltered,
            spike_template_width=stw,
            fs=fs,
        )

        assert len(result.dtw_distances) == 0
        assert len(result.amplitudes) == 0

    def test_dtw_distances_nonnegative(self, spike_template, default_params):
        fs = default_params.fs
        stw = default_params.spike_template_width
        n_samples = 15000
        rng = np.random.default_rng(42)

        filtered = rng.normal(0, 0.01, n_samples)
        unfiltered = rng.normal(0, 0.01, n_samples)

        spike_locs = np.array([4000, 9000], dtype=np.int64)
        half = stw // 2
        for loc in spike_locs:
            filtered[loc - half:loc + half + 1] += spike_template * 10

        result = match_template(
            spike_locs=spike_locs,
            spike_template=spike_template,
            filtered_data=filtered,
            unfiltered_data=unfiltered,
            spike_template_width=stw,
            fs=fs,
        )

        assert np.all(result.dtw_distances >= 0)
