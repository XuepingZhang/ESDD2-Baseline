import torch.nn as nn
from .Unet_mask import UNetSTFTComplexRefine
from conf import aasist_conf, unet_conf
from .XLSR2_AASIST import XLSR2_AASIST


class Model(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.spar = UNetSTFTComplexRefine(unet_base_ch=unet_conf['unet_base_ch'], unet_layers=unet_conf['unet_layers'])


        self.aasist_original = XLSR2_AASIST(aasist_conf)
        self.aasist_speech = XLSR2_AASIST(aasist_conf)
        self.aasist_env = XLSR2_AASIST(aasist_conf)


    def forward(self, egs):
        h_original, res_original = self.aasist_original(egs['mix'])
        speech_, env_ = self.spar(egs['mix'])

        h_speech_, res_speech_ = self.aasist_speech(speech_)
        h_env_, res_env_ = self.aasist_env(env_)
        h_speech, res_speech = self.aasist_speech(egs["ref"][0])
        h_env, res_env = self.aasist_env(egs["ref"][1])

        return speech_, env_\
            , res_speech_, res_env_, res_speech, res_env, res_original\
            , h_original, h_speech_, h_env_, h_speech, h_env


