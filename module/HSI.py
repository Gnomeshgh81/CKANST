import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper class for Instance Normalization with learnable affine parameters
# The paper denotes "Norm" in the diagrams, which is typically InstanceNorm in AST.
class ResIN(nn.Module):
    def __init__(self, num_features, affine=True):
        super(ResIN, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=affine)
    def forward(self, x):
        return self.norm(x)

# GlobalStyleAggregation module
class GlobalStyleAggregation(nn.Module):
    def __init__(self, in_channels):
        super(GlobalStyleAggregation, self).__init__()
        self.in_channels = in_channels

        self.dynamic_net_ks = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0)
        )
        self.dynamic_net_vs = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, 1, 1, 0)
        )

    def calculate_statistics(self, feature_map):
        """
        Calculates channel-wise mean, standard deviation, skewness, and kurtosis.
        Args:
            feature_map (torch.Tensor): Input feature map (B, C, H, W).
        Returns:
            tuple: (mean, std, skewness, kurtosis), each (B, C, 1, 1).
        """
        B, C, H, W = feature_map.shape
        # Flatten spatial dimensions for easier calculation of moments
        flat_feature_map = feature_map.view(B, C, -1) # (B, C, H*W)

        # Mean (μ)
        mean = torch.mean(flat_feature_map, dim=2, keepdim=True).unsqueeze(-1) # (B, C, 1, 1)

        # Centered features (x - μ)
        # unsqueeze(-1) for broadcasting mean to H*W for element-wise operation
        centered_feature_map = flat_feature_map - mean.squeeze(-1)

        # Variance (σ^2)
        variance = torch.mean(centered_feature_map.pow(2), dim=2, keepdim=True).unsqueeze(-1) # (B, C, 1, 1)

        # Standard deviation (σ), adding epsilon for numerical stability
        std = torch.sqrt(variance + 1e-5) # (B, C, 1, 1)

        # Normalized features ((x - μ) / σ)
        # unsqueeze(-1) for broadcasting std to H*W for element-wise operation
        normalized_features = centered_feature_map / std.squeeze(-1)

        skewness = torch.mean(normalized_features.pow(3), dim=2, keepdim=True).unsqueeze(-1) # (B, C, 1, 1)

        kurtosis = torch.mean(normalized_features.pow(4), dim=2, keepdim=True).unsqueeze(-1) # (B, C, 1, 1)

        return mean, std, skewness, kurtosis

    def forward(self, style_k, style_v):
        """
        Processes style_k (from the style feature map) to generate global style vectors Ks and Vs.
        Args:
            style_k (torch.Tensor): Key features from style (B, C, H, W).
            style_v (torch.Tensor): Value features from style (B, C, H, W).
                                    (Note: style_v is not directly used for statistics, but could be for other interpretations.)
        Returns:
            tuple: (ks, vs), global style vectors, each (B, C, 1, 1).
        """
        # Calculate statistics from style_k, which represents the style patterns
        mean, std, skewness, kurtosis = self.calculate_statistics(style_k)

        # Concatenate the four channel-wise statistics along the channel dimension
        # Resulting shape: (B, 4 * C, 1, 1)
        stats = torch.cat([mean, std, skewness, kurtosis], dim=1)

        # Generate global style representation vectors Ks and Vs using dynamic networks
        ks = self.dynamic_net_ks(stats) # (B, C, 1, 1)
        vs = self.dynamic_net_vs(stats) # (B, C, 1, 1)

        return ks, vs

# Holistic Style Injector (HSI) module
class HSI(nn.Module):
    def __init__(self, in_channels):
        super(HSI, self).__init__()
        self.in_channels = in_channels

        # Q, K, V transformations: Normalization (InstanceNorm) followed by 1x1 Conv
        self.norm_q = ResIN(in_channels)
        self.conv_q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        self.norm_k = ResIN(in_channels)
        self.conv_k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        self.norm_v = ResIN(in_channels)
        self.conv_v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        # Module to aggregate global style information (Figure 4)
        self.global_style_aggregation = GlobalStyleAggregation(in_channels)

        # Output convolution for the transformed feature O (Figure 3b)
        self.conv_o = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        # Sigmoid activation for the dynamic weighting factor Ag
        self.sigmoid = nn.Sigmoid()
        self.eps = 1e-6

    def _lambda_g(self, Qc, Ks):
        """
        Cosine similarity between global content (Qc) and global style (Ks),
        mapped to [0,1] by (cos + 1)/2. Returns (B,1,1,1).
        """
        # both (B,C,1,1) -> (B,C)
        q = Qc.flatten(1)
        k = Ks.flatten(1)
        num = (q * k).sum(dim=1, keepdim=True)  # (B,1)
        den = (q.norm(dim=1, keepdim=True) * k.norm(dim=1, keepdim=True)).clamp_min(self.eps)
        cos = (num / den)  # (B,1)
        lam = 0.5 * (cos + 1.0)  # map to [0,1]
        return lam.view(-1, 1, 1, 1)  # (B,1,1,1)

    def forward(self, content_features, style_features):
        """
        Applies holistic style injection to content features based on style features.
        Args:
            content_features (torch.Tensor): Content feature map (B, C, H, W).
            style_features (torch.Tensor): Style feature map (B, C, H, W).
        Returns:
            torch.Tensor: Stylized feature map (Fcs) (B, C, H, W).
        """
        # 1. Compute Q, K, V from content and style features

        t = self.norm_q(content_features)
        q = self.conv_q(t)  # Query from content
        k = self.conv_k(self.norm_k(style_features))   # Key from style
        v = self.conv_v(style_features)   # Value from style


        # 2. Global Content Aggregation (Qc)
        # Average pooling to get a 1x1xC representation of content query
        qc = F.adaptive_avg_pool2d(q, (1, 1)) # (B, C, 1, 1)

        # 3. Global Style Aggregation (Ks, Vs)
        # Uses the custom GSA module to derive Ks and Vs from k and v
        ks, vs = self.global_style_aggregation(k, v) # ks, vs are (B, C, 1, 1)

        # 4. Dynamic Dual Relation Construction (Equation 8 and Figure 3b)
        # Ag = Sigmoid(Qc ⊙ Ks) - dynamic weighting factor for global relations
        ag = self._lambda_g(q , k) # Element-wise multiplication, (B, C, 1, 1)

        # Global-content-to-global-style interaction (Qc ⊙ Ks)
        global_global_relation = qc * ks # (B, C, 1, 1)

        # Local-content-to-global-style interaction (Q ⊙ Ks)
        # Ks (B,C,1,1) is broadcasted to (B,C,H,W) for element-wise multiplication with Q (B,C,H,W)
        local_global_relation = q * ks # (B, C, H, W)

        # Fqk = Ag ⊙ (Qc ⊙ Ks) + (1 - Ag) ⊙ (Q ⊙ Ks)
        # Ag (B, C, 1, 1) is broadcasted for weighting local_global_relation
        f_qk = ag * global_global_relation + (1 - ag) * local_global_relation
        # f_qk has shape (B, C, H, W)
        A = self.sigmoid(f_qk)

        # 5. Final Output O
        # O is derived from Fqk via a 1x1 convolution
        o = self.conv_o(A * vs) # (B, C, H, W)

        # 6. Residual connection: Fcs = O + content_features
        # This preserves content structure
        fcs = o + t

        return fcs



# Example Usage:
if __name__ == '__main__':
    batch_size = 2
    channels = 256 # A common number of channels for VGG features
    height, width = 64, 64 # Spatial resolution of feature maps

    # Dummy input features
    content_features = torch.randn(batch_size, channels, height, width)
    style_features = torch.randn(batch_size, channels, height, width)

    print(f"Input content features shape: {content_features.shape}")
    print(f"Input style features shape: {style_features.shape}")

    # Initialize the HSI module
    hsi_module = HSI(in_channels=channels)

    # Perform a forward pass
    output_features = hsi_module(content_features, style_features)

    print(f"Output features shape (Fcs): {output_features.shape}")

    # Assert that the output shape matches the input content feature shape
    assert output_features.shape == content_features.shape
    print("Shape assertion passed!")

    # Verify GlobalStyleAggregation separately if needed
    gsa_module = GlobalStyleAggregation(in_channels=channels)
    style_k_dummy = torch.randn(batch_size, channels, height, width)
    style_v_dummy = torch.randn(batch_size, channels, height, width)
    ks_out, vs_out = gsa_module(style_k_dummy, style_v_dummy)
    print(f"Ks shape from GSA: {ks_out.shape}")
    print(f"Vs shape from GSA: {vs_out.shape}")
    assert ks_out.shape == (batch_size, channels, 1, 1)
    assert vs_out.shape == (batch_size, channels, 1, 1)
    print("GlobalStyleAggregation shape assertion passed!")

    # Test with a different resolution
    height, width = 128, 128
    content_features_large = torch.randn(batch_size, channels, height, width)
    style_features_large = torch.randn(batch_size, channels, height, width)
    output_features_large = hsi_module(content_features_large, style_features_large)
    print(f"Output features shape (128x128): {output_features_large.shape}")
    assert output_features_large.shape == content_features_large.shape
    print("Shape assertion for 128x128 passed!")

    # Test with a very large resolution for linear complexity check (without actual GPU measurement)
    # The module should handle it without quadratic complexity issues for memory
    height, width = 1024, 1024
    content_features_vlarge = torch.randn(batch_size, channels, height, width)
    style_features_vlarge = torch.randn(batch_size, channels, height, width)
    output_features_vlarge = hsi_module(content_features_vlarge, style_features_vlarge)
    print(f"Output features shape (1024x1024): {output_features_vlarge.shape}")
    assert output_features_vlarge.shape == content_features_vlarge.shape
    print("Shape assertion for 1024x1024 passed!")