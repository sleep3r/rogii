"""
dlmtp  —  DL-MTP: GR/typewell heatmap + U-Net + DP decoder
============================================================

Idea: treat TVT prediction as a ridge-finding problem in a 2D image.

For each hidden row s, build a TVT grid centred on the K3 prior:
    t_grid[s, j] = prior_tvt[s] + (j - J//2) * bin_ft

The heatmap encodes how well the horizontal GR matches the typewell GR
sampled at each candidate TVT value.  A U-Net predicts logits over bins;
a DP (Viterbi) decoder finds a smooth, physically consistent path.

Run 1 target: DP RMSE < K3/local-GR baselines (< 12 ft).
"""
__version__ = "0.1.0"
