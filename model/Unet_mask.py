# unet_stft_complex_refine.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        x = F.pad(x, [0, diff_x, 0, diff_y])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.proj(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, num_layers=4, out_ch=2):
        super().__init__()
        chs = [base_ch*(2**i) for i in range(num_layers)]
        self.inc = DoubleConv(in_ch, chs[0])
        self.down = nn.ModuleList([Down(chs[i], chs[i+1]) for i in range(num_layers-1)])
        self.up = nn.ModuleList([Up(chs[i], chs[i-1]) for i in range(num_layers-1,0,-1)])
        self.outc = OutConv(chs[0], out_ch)
    def forward(self, x):
        xs = [self.inc(x)]
        for down in self.down:
            xs.append(down(xs[-1]))
        y = xs[-1]
        for i, up in enumerate(self.up):
            skip = xs[-2-i]
            y = up(y, skip)
        return self.outc(y)


class UNetSTFTComplexRefine(nn.Module):
    def __init__(self,
                 n_fft=1024, hop_length=256, win_length=1024, center=True,
                 unet_base_ch=32, unet_layers=3):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center

        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window, persistent=False)

        self.unet_speech = UNet2D(in_ch=1, base_ch=unet_base_ch, num_layers=unet_layers, out_ch=2)



    def _pad_to_factor(self, x, factor_h, factor_w):
        if not torch.is_tensor(x):
            raise TypeError(f"x should be a tensor, got {type(x)}")
        B, C, freq, T = x.shape
        pad_h = (factor_h - freq % factor_h) % factor_h
        pad_w = (factor_w - T % factor_w) % factor_w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (freq, T)


    def _crop_to_size(self, x, size_hw):
        F,T = size_hw
        return x[..., :F, :T]

    def forward(self, waveform):
        B, L = waveform.shape

        # 1) STFT
        stft = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=self.center, return_complex=True
        )  # [B,F,T]

        mag = stft.abs().unsqueeze(1)  # [B,1,F,T]

        factor = 2 ** len(self.unet_speech.down)
        mag_pad, orig_size = self._pad_to_factor(mag, factor, factor)

        # 2) speech mask
        mask_speech_pad = self.unet_speech(mag_pad)  # [B,2,F_pad,T_pad]
        mask_speech_pad = self._crop_to_size(mask_speech_pad, orig_size)
        re_mask, im_mask = mask_speech_pad[:, 0], mask_speech_pad[:, 1]
        complex_mask = torch.complex(re_mask, im_mask)
        est_speech_stft = complex_mask * stft.unsqueeze(1)[:, 0, ...]
        speech_waveform = torch.istft(est_speech_stft, n_fft=self.n_fft,
                                      hop_length=self.hop_length, win_length=self.win_length,
                                      window=self.window, center=self.center, length=L)

        # 3) residual waveform

        residual_waveform = (waveform - speech_waveform).detach()

        # 4) STFT of residual
        residual_stft = torch.stft(
            residual_waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=True, return_complex=True
        )

        # 5) adaptive soft-mask in frequency domain
        speech_mag = est_speech_stft.abs().detach()
        residual_mag = residual_stft.abs()

        # scale_factor = mean(residual_mag) / (mean(speech_mag) + eps)
        eps = 1e-8
        scale_factor = residual_mag.mean(dim=(1, 2), keepdim=True) / (speech_mag.mean(dim=(1, 2), keepdim=True) + eps)

        freq_mask = 1 - torch.tanh(speech_mag / (residual_mag + eps) * scale_factor)

        # 6) apply mask
        enhanced_residual_stft = residual_stft * freq_mask

        # 7) ISTFT to waveform
        bg_waveform = torch.istft(enhanced_residual_stft, n_fft=self.n_fft,
                                  hop_length=self.hop_length, win_length=self.win_length,
                                  window=self.window, center=True, length=L)

        return speech_waveform, bg_waveform



if __name__ == "__main__":
    torch.manual_seed(0)
    B,L = 2,64000
    x = torch.randn(B,L)
    model = UNetSTFTComplexRefine()
    with torch.no_grad():
        speech, bg = model(x)
        print("input:", x.shape, "speech:", speech.shape, "bg:", bg.shape)
