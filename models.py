import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    # Consists of 3x3 Conv -> ReLU -> 2x2 MaxPool
    def __init__(self, in_chans, out_chans, downsample=2, padding="same"):    
        """
        Parameters:
            in_chans: number of channels in the input 
            out_chans: number of channels in the output
            upsample: downsampling factor. Same as upsampling factor in DecoderBlock
            padding: padding applied to Conv2d
        """
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, 1, padding=padding)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(downsample)
    def forward(self, x):
        """
        Input:
            x: input feature map of size (B, C_in, H, W)
        Returns:
            output feature map of size (B, C_out, H//downsample, W//downsample)
        """
        conv_out = self.relu(self.conv(x))
        mp_out = self.mp(conv_out)
        return mp_out, conv_out

class DecoderBlock(nn.Module):
    # Consists of interpolate -> 3x3 conv -> concat (if allowed) -> 3x3 Conv -> relu
    def __init__(self, in_chans, out_chans, skip=True, upsample=2, padding="same"):
        """
        Parameters:
            in_chans: number of channels in the input 
            out_chans: number of channels in the output
            skip: whether or not to have skip connections
            upsample: upsampling factor. Same as downsampling factor
            padding: padding applied to Conv2d
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, in_chans, 3, 1, padding=padding)
        self.relu1 = nn.ReLU()
        
        # If concatentate then grow input dimension by 2
        skip_factor = 2 if skip else 1
        self.conv2 = nn.Conv2d(in_chans*skip_factor, out_chans, 3, 1, padding=padding)
        self.relu2 = nn.ReLU()
        self.upsample = upsample
        self.skip = skip
        self.padding = padding
    def forward(self, x, enc_features=None):
        """
        Input:
            x: input feature map of size (B, C_in, H, W)
            enc_feature: input feature from EncoderBlock if skip connection is used
        Returns:
            output feature map of size (B, C_out, H*upsample, W*upsample)
        """
        x = nn.functional.interpolate(x, scale_factor=self.upsample, mode="bilinear")
        x = self.relu1(self.conv1(x))
        if self.skip:
            if self.padding != "same":
                # Crop the enc_features to the same size as input
                w = x.size(-1)
                c = (enc_features.size(-1) - w) // 2
                enc_features = enc_features[:,:,c:c+w,c:c+w]
            x = torch.cat((enc_features, x), dim=1)
        x = self.relu2(self.conv2(x))
        return x

class UNet(nn.Module):
    def __init__(self, nclass=1, in_chans=3, depth=3, skip=True, sample_factor=2, padding="same"):
        """
        Parameters:
            nclass: number of class
            in_chans: number of channels in the input
            depth: depth of the U-Net
            skip: whether or not to have skip connections
            sample_factor: upsampling & downsampling factor
            padding: padding applied to Conv2d
        """
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        out_chans = 16
        for _ in range(depth):
            self.encoder.append(EncoderBlock(in_chans, out_chans, downsample=sample_factor, padding=padding))
            in_chans, out_chans = out_chans, out_chans*2

        # Fully convolution layer that connects encoder and decoder
        self.fc = nn.Conv2d(in_chans, in_chans, 3, 1, padding=padding)

        out_chans = in_chans // 2
        for _ in range(depth-1):
            self.decoder.append(DecoderBlock(in_chans, out_chans, skip, upsample=sample_factor, padding=padding))
            in_chans, out_chans = out_chans, out_chans//2
        self.decoder.append(DecoderBlock(in_chans, in_chans, skip, upsample=sample_factor, padding=padding))
        # Add a 1x1 convolution to produce final classes
        self.logits = nn.Conv2d(in_chans, nclass, 1, 1)

    def forward(self, x):
        encoded = []
        for enc in self.encoder:
            x, enc_output = enc(x)
            encoded.append(enc_output)
        x = self.fc(x)
        for dec in self.decoder:
            enc_output = encoded.pop()
            x = dec(x, enc_output)

        # Return the logits
        return self.logits(x)