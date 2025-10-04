import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class Residual3DBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, p_dim: int = 64):
        super().__init__()

        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels + p_dim, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor, p_i: torch.Tensor):


        residual = self.shortcut(x)

        out_1 = F.relu(self.bn1(self.conv1(x)))

        p_expanded = p_i.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, x.size(2), x.size(3), x.size(4))

        out_1_with_p = torch.cat([out_1, p_expanded], dim=1)

        out = self.bn2(self.conv2(out_1_with_p))

        out +=  residual
        out = F.relu(out)

        return out

class Encoder(nn.Module):

    def __init__(self, in_channels: int, base_channels: int = 32, p_dim: int = 64):
        super().__init__()
        self.stage1 = Residual3DBlock(in_channels, base_channels, p_dim)
        self.downsample1 = nn.Conv3d(base_channels, base_channels*2, kernel_size=2, stride=2)
        
        self.stage2 = Residual3DBlock(base_channels*2, base_channels*2, p_dim)
        self.downsample2 = nn.Conv3d(base_channels*2, base_channels*4, kernel_size=2, stride=2)
        
        self.stage3 = Residual3DBlock(base_channels*4, base_channels*4, p_dim)
        self.downsample3 = nn.Conv3d(base_channels*4, base_channels*8, kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor, p_i: torch.Tensor) -> List[torch.Tensor]:
        #
        s1 = self.stage1(x, p_i)
        d1 = self.downsample1(s1)
        
        #
        s2 = self.stage2(d1, p_i)
        d2 = self.downsample2(s2)
        
        #
        s3 = self.stage3(d2, p_i)
        d3 = self.downsample3(s3)
        
        return [s1, s2, s3, d3]  # return multi-layer feature

class MirrorDecoder(nn.Module):
    """"""
    def __init__(self, base_channels: int = 32, p_dim: int = 64):
        super().__init__()

        self.upsample0 = nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=2, stride=2)  # [1,512,1,1,1]->[1,256,2,2,2]
        self.stage0 = Residual3DBlock(base_channels * 4, base_channels * 4, p_dim)  # [1,256,2,2,2] -> [1,256,2,2,2]

        # skip connection
        self.upsample1 = nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=2, stride=2) # [1,512,2,2,2] --> [1,256,4,4,4]
        self.stage1 = Residual3DBlock(base_channels*4, base_channels*2, p_dim)    # [1,256,4,4,4] --> [1,128,4,4,4]

        # skip connection
        self.upsample2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=2, stride=2)  # [1,256,4,4 ,4] -->  [1,128,8,8,8]
        self.stage2 = Residual3DBlock(base_channels*2, base_channels, p_dim)   #[1,128,8,8,8] --> [1,64,8,8,8]

        # skip connection
        self.stage3 = Residual3DBlock(base_channels*2, base_channels, p_dim)  # [1,128,8,8,8]  --> [1,64,8,8,8]

        self.final_conv = nn.Conv3d(base_channels, base_channels, kernel_size=1)     # [1,64,8,8,8] --> [1,64,8,8,8]
        
    def forward(self, features: List[torch.Tensor], p_i: torch.Tensor) -> torch.Tensor:
        s1, s2, s3, d3 = features

        u0 = self.upsample0(d3)
        u0 = self.stage0(u0, p_i)

        u1 = torch.cat([u0, s3], dim=1)
        u1 = self.upsample1(u1)

        u1 = self.stage1(u1, p_i)

        u2 = torch.cat([u1, s2], dim=1)
        u2 = self.upsample2(u2)
        u2 = self.stage2(u2, p_i)

        u3 = torch.cat([u2, s1], dim=1)
        u3 = self.stage3(u3, p_i)
        
        return self.final_conv(u3)

class G_Model(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, p_dim: int = 64, num_modality=4):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, p_dim)
        self.decoder = MirrorDecoder(base_channels, p_dim)
        self.modality_embeddings = nn.Parameter(torch.randn(num_modality, p_dim))
        
    def forward(self, x_i: torch.Tensor, modality_id: torch.Tensor) -> torch.Tensor:
        p_i = self.modality_embeddings[modality_id]
        p_i = p_i.unsqueeze(0).expand(x_i.size(0), -1)

        features = self.encoder(x_i, p_i)
        out = self.decoder(features, p_i)
        return out

class CyclicSynthesis(nn.Module):

    def __init__(self, num_modalities: int = 4, in_channels: int = 1, base_channels: int = 32, p_dim: int = 64):
        super().__init__()
        self.M = num_modalities
        self.generators = nn.ModuleList([
            G_Model(in_channels, base_channels, p_dim)
            for _ in range(num_modalities)
        ])


        
    def forward(self, z_i: torch.Tensor, current_modality: int) -> torch.Tensor:

        p_i = self.modality_embeddings[current_modality]
        p_i = p_i.unsqueeze(0).expand(z_i.size(0), -1)  # 扩展到batch size
        

        next_modality = (current_modality + 1) % self.M
        return self.generators[current_modality](z_i, p_i), next_modality
    
    def cyclic_generation(self, initial_input: torch.Tensor, start_modality: int = 0) -> List[torch.Tensor]:

        results = []
        current_z = initial_input
        current_modality = start_modality
        
        for _ in range(self.M):
            next_z, next_modality = self.forward(current_z, current_modality)
            results.append(next_z)
            current_z = next_z
            current_modality = next_modality
            
        return results

def test_encoder():
    enc_feat = torch.randn(1, 64, 8, 8, 8)
    net = Encoder(in_channels=64, base_channels=64, p_dim=64)

    prompt_vec = torch.randn(1,64)
    out = net(enc_feat,prompt_vec)

    for item in out:
        print(item.shape)
#     torch.Size([1, 64, 8, 8, 8])
        # torch.Size([1, 128, 4, 4, 4])
        # torch.Size([1, 256, 2, 2, 2])
        # torch.Size([1, 512, 1, 1, 1])

def test_decoder():
    in_chns = 64
    enc_feat = torch.randn(1, in_chns, 16, 16, 16)

    p_dim = 64
    prompt_vec = torch.randn(1,p_dim)
    net = Encoder(in_channels=in_chns, base_channels=in_chns, p_dim=p_dim)


    out_list = net(enc_feat,prompt_vec)
    print("encoder finished ")
    decoder = MirrorDecoder(in_chns, p_dim=p_dim)
    out = decoder(out_list, prompt_vec)
    print(out.shape)
    print("decoder finished ")


def main():
    in_chns = 64
    framework = CyclicSynthesis(num_modalities=4, in_channels=in_chns, base_channels=in_chns)

    #  [batch, channels, depth, height, width]
    batch_size = 1

    input_data = torch.randn(batch_size, in_chns, 32, 32, 32)

    #
    output, next_modality = framework(z_i=input_data, current_modality=0)
    print(f"Generated modality {next_modality}, output shape: {output.shape}")

    #
    all_outputs = framework.cyclic_generation(input_data, 0)
    print(f"Cyclic generation produced {len(all_outputs)} modalities")

if __name__ == "__main__":
    pass
    # test_encoder()
    # test_decoder()
    main()