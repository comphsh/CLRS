import os
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from networks.module.encoder_and_decoder import Downsampling, LayerNormGeneral, Mlp, Block , Encoder , SepConv ,EfficientAttention
from monai.networks.blocks import UnetOutBlock

downlayer_fusion_partial = [partial(Downsampling,
         kernel_size=3, stride=1, padding=1,
         post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),  pre_permute=True
         )]

def Norm_layer(normalization, n_filters_out):
    if normalization == 'batchnorm':
        out = nn.BatchNorm3d(n_filters_out)
    elif normalization == 'groupnorm':
        out = nn.GroupNorm(num_groups=16, num_channels=n_filters_out)
    elif normalization == 'instancenorm':
        out = nn.InstanceNorm3d(n_filters_out, affine=True)

    elif normalization == 'Syncbatchnorm':
        out = nn.SyncBatchNorm(n_filters_out)
    elif normalization == 'layernorm':
        out = nn.LayerNorm(n_filters_out)
    elif normalization != 'none':
        assert False
    return out

def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class SpecificClassification(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0 , batch_size=1):
        super(SpecificClassification, self).__init__()
        self.has_dropout = has_dropout

        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, padding=0, stride=1),
            Norm_layer(normalization=normalization,n_filters_out=n_channels),
            nn.AdaptiveMaxPool3d(1),
        )

        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, padding=0, stride=1),
            Norm_layer(normalization=normalization, n_filters_out=n_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, padding=0, stride=1),
            Norm_layer(normalization=normalization, n_filters_out=n_channels),
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, padding=0, stride=1),
            Norm_layer(normalization=normalization, n_filters_out=n_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, padding=0, stride=1),
            Norm_layer(normalization=normalization, n_filters_out=n_channels),
        )

        self.branch4 = nn.Sequential(
            nn.Conv3d(in_channels=n_channels, out_channels=n_channels, kernel_size=1, padding=0, stride=1),
            Norm_layer(normalization=normalization, n_filters_out=n_channels),
            nn.AdaptiveAvgPool3d(1),
        )

        self.maxpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels * batch_size , 512) ,
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )
        self.dropout = nn.Dropout1d(p=0.1, inplace=False)

        self.cls_head = nn.Linear(512, n_classes)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, features):

        ft1 = self.branch1(features)

        ft2 = self.branch2(features)

        ft3 = self.branch3(features)

        ft4 = self.branch4(features)

        max_fuse_fm = ft2 * ft1

        avg_fuse_fm = ft3 * ft4

        add_fuse_fm = max_fuse_fm + avg_fuse_fm


        max_fm = self.maxpool(add_fuse_fm)
        max_fm = max_fm.flatten()

        proj_feature = self.fc(max_fm)


        if self.has_dropout == True:
            proj_feature = self.dropout(proj_feature)

        cls_feature = self.cls_head(proj_feature)

        return cls_feature, add_fuse_fm, proj_feature

def get_fusion_transform(in_chns , out_chns):
    downsample_layers = nn.ModuleList([downlayer_fusion_partial[0](in_chns, out_chns) ])
    return downsample_layers[0]

def get_spec_transform(n_modality, convtype=''):
    # ---------- set spec conv ----------------
    depths = 2
    dims = 256
    if convtype == 'sepconv':
        token_mixers = SepConv
    elif convtype == 'efficient_attention':
        token_mixers = EfficientAttention
    else:
        raise ValueError("not found conv type")

    mlps = Mlp
    drop_path_rate = 0.1
    layer_scale_init_values = None
    res_scale_init_values = None
    position_embeddings = [None, None]
    norm_layers = partial(LayerNormGeneral, eps=1e-6, bias=False)

    dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
    # ---------------- ----------------
    # cal divide diffence modality vec
    # [(0, 64), (64, 128), (128, 192), (192, 255)]
    modality_vec_list = split_into_list_general(dims, n_modality)

    # block require torch.Size([3, 16, 16, 16, 256(channel)]) to match layer
    # torch.Size([3, 16, 16, 16, 256])
    spec_map = nn.Sequential(
        *[Block(dim=dims,
                token_mixer=token_mixers,
                mlp=mlps,
                norm_layer=norm_layers,
                drop_path=dp_rates[0+j],
                layer_scale_init_value=layer_scale_init_values,
                res_scale_init_value=res_scale_init_values,
                position_embedding=position_embeddings[j],
                ) for j in range(depths)]
    )


    return spec_map, modality_vec_list


class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_modality=4, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(Encoder, self).__init__()
        # [80,128,128]->[80,128,128]
        self.pre_chns = n_filters
        self.pre_procross = nn.Sequential(
            nn.Conv3d(n_channels, self.pre_chns, kernel_size=7, stride=(1, 1, 1), padding=3, bias=False),
            nn.BatchNorm3d(self.pre_chns),
            nn.ReLU(inplace=True),
            # [(i-3) + 2*pad] / 2 + 1
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
        )

        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, self.pre_chns, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):


        input = self.pre_procross(input)


        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res

class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2,n_modality=4,  n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        # self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv = nn.Identity()

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return out_seg

def split_into_list_general(total_size, num_parts):
    part_size = total_size // num_parts
    remainder = total_size % num_parts
    result = []
    start = 0
    for i in range(num_parts):
        end = start + part_size
        if i < remainder:
            end += 1  # Distribute the remainder
        result.append((start, end))
        start = end
    return result




class CLRSNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_modality=4, n_filters=16, normalization='none', spec_normalization='none' ,has_dropout=False,
                 has_residual=False, args=None,spatial_dims=3):
        super(CLRSNet, self).__init__()
        self.n_modality = args.num_modality

        self.n_modality = n_modality
        self.encoder = Encoder(n_channels, n_classes, n_modality, n_filters, normalization, has_dropout, has_residual)
        self.decoder = Decoder(n_channels, n_classes, n_modality , n_filters, normalization, has_dropout, has_residual, 0)


        self.spec_module_list = nn.ModuleList()


        self.spec_transform, self.modality_vec_list = get_spec_transform(args.num_modality, convtype='sepconv')
        self.spec_transform02, _ = get_spec_transform(args.num_modality, convtype='efficient_attention')


        self.fusion_transform = get_fusion_transform(512, 256)
        for iij in range(args.num_modality):
            chns_len = self.modality_vec_list[iij][1] - self.modality_vec_list[iij][0]
            spec_module1 = SpecificClassification(n_channels=chns_len, n_classes=args.num_modality, batch_size=args.batch_size, normalization=args.spec_normalization).cuda()
            self.spec_module_list.append(spec_module1)


        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')


        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=n_filters,
            out_channels=n_classes,
        )


    def forward(self, data):
        features = self.encoder(data)
        # 0 torch.Size([1, 16, 128, 128, 128])
        # 1 torch.Size([1, 32, 64, 64, 64])
        # 2 torch.Size([1, 64, 32, 32, 32])
        # 3 torch.Size([1, 128, 16, 16, 16])
        # 4 torch.Size([1, 256, 8, 8, 8])

        # for item in features:
        #     print(item.shape)

        spec_fm_cat = None
        vec_list = []
        proj_feature_list = []
        for iij in range(self.n_modality): # 0, 1, 2, 3 or 0,1
            vec1, add_fuse_fm1, proj_feature1 = self.spec_module_list[iij](features[4][iij, :, :, :].unsqueeze(0))
            vec_list.append(vec1)
            proj_feature_list.append(proj_feature1)

            if spec_fm_cat == None:
                spec_fm_cat = add_fuse_fm1
            else:
                spec_fm_cat = torch.cat([spec_fm_cat, add_fuse_fm1], dim=0).cuda()



        bs_modaln, _ , _, _, _ = features[0].shape
        features_view = [i_layer.view(bs_modaln // self.n_modality , self.n_modality * i_layer.shape[1], i_layer.shape[2], i_layer.shape[3], i_layer.shape[4]) for  i_layer in features]


        spec_fm_cat_view = spec_fm_cat.view(bs_modaln // self.n_modality , self.n_modality* spec_fm_cat.shape[1] ,  spec_fm_cat.shape[2], spec_fm_cat.shape[3], spec_fm_cat.shape[4])

        logits = self.decoder(features=features_view , spec_fm_cat=spec_fm_cat_view)
        logits = self.upsample(logits)

        spec_logtis_cat = torch.stack(vec_list , dim=0).cuda()

        spec_info_vector = torch.stack(proj_feature_list, dim=0).cuda()

        return [logits , spec_logtis_cat , spec_info_vector, spec_fm_cat]

    def forward_encoder(self, x):

        self.bs, C, _, _, _ = x.shape
        assert C == self.n_modality

        features_list = self.encoder(x)
        confused_ft = features_list[-1]


        data1 = confused_ft.permute(0, 2, 3, 4, 1)
        spec_map = self.spec_transform(data1).permute(0, 4, 1, 2, 3)


        spec_fm_cat = None
        logit_list = []
        proj_feature_list = []


        generated_map = []
        for iij in range(self.n_modality):


            beginx = self.modality_vec_list[iij][0]
            endx = self.modality_vec_list[iij][1]

            # for synthesis
            single_modal = spec_map[:, beginx:endx, ...]
            generated_map.append(single_modal)
            logit1, add_fuse_fm1, proj_feature1 = self.spec_module_list[iij](single_modal)



            logit_list.append(logit1)
            proj_feature_list.append(proj_feature1)
            if spec_fm_cat == None:
                spec_fm_cat = add_fuse_fm1
            else:
                spec_fm_cat = torch.cat([spec_fm_cat, add_fuse_fm1], dim=1).cuda()



        spec_fm_cat = self.spec_transform02(spec_fm_cat.permute(0, 2, 3, 4, 1))


        fuse_map = self.fusion_transform(torch.cat([features_list[-1].permute(0,2,3,4,1) , spec_fm_cat ] , dim=-1))



        features_list[-1] = features_list[-1]  + fuse_map.permute(0,4,1,2,3)



        spec_logtis_cat = torch.stack(logit_list, dim=0).cuda()
        spec_info_vector = torch.stack(proj_feature_list,  dim=0).cuda()

        return [features_list, generated_map, spec_fm_cat, spec_logtis_cat, spec_info_vector]

    def forward_decoder(self, features_list):
        dec_out = self.decoder(features_list)
        out = self.out(dec_out)
        return [out]



def cal_model_complexity_by_summary():
    import argparse
    parser = argparse.ArgumentParser()  ## parse arguments
    args = parser.parse_args()

    setattr(args, 'seed', 42)
    setattr(args, 'num_modality', 4)
    setattr(args, 'batch_size', 1)
    setattr(args, 'spec_normalization', "batchnorm")

    model = CLRSNet(n_channels=4, n_classes=3, n_modality=args.num_modality, normalization='batchnorm', has_dropout=False, args=args).cuda()

    data = torch.randn((1,4,128,128,128)).cuda()
    for i in range(10):
        features_list, generated_map, spec_fm_cat, spec_logtis_cat, spec_info_vector = model.forward_encoder(data)
        pred_obj = model.forward_decoder(features_list)
        logits = pred_obj[0]
        print(logits.shape)
        exit()

    print(logits.shape , logits.unique())

if __name__ == '__main__':
    pass
    # cal_model_complexity_by_summary()

    # ipdb.set_trace()
