import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from model.vit import forward_flex
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)
from model.utils import _make_fusion_block, norm_normalize, sample_points


class DPT_Uncertain(BaseModel):
    def __init__(
        self,
        head,
        sampling_ratio=None,
        importance_ratio=None,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT_Uncertain, self).__init__()

        self.sampling_ratio = sampling_ratio
        self.importance_ratio = importance_ratio

        
        # produces 1/8 res output
        # 512 -> 128
        self.out_conv_res8 = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

        # produces 1/4 res output
        self.out_conv_res4 = nn.Sequential(
            nn.Conv1d(256 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

        # produces 1/2 res output
        self.out_conv_res2 = nn.Sequential(
            nn.Conv1d(256 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

        # produces 1/1 res output
        self.out_conv_res1 = nn.Sequential(
            nn.Conv1d(256 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x, gt_norm_mask=None, mode='train'):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # 1/8 res
        out_res8 = self.out_conv_res8(path_3)     # out_res8: [2, 4, 60, 80]      1/8 res output
        out_res8 = norm_normalize(out_res8)       # out_res8: [2, 4, 60, 80]      1/8 res output       

        # 1/4 res
        if mode == 'train':
            # upsampling ... out_res8: [2, 4, 60, 80] -> out_res8_res4: [2, 4, 120, 160]
            out_res8_res4 = F.interpolate(out_res8, scale_factor=2, mode='bilinear', align_corners=True)
            B, _, H, W = out_res8_res4.shape

            # samples: [B, 1, N, 2]
            point_coords_res4, rows_int, cols_int = sample_points(out_res8_res4.detach(), gt_norm_mask,
                                                                  sampling_ratio=self.sampling_ratio,
                                                                  beta=self.importance_ratio)

            # output (needed for evaluation / visualization)
            out_res4 = out_res8_res4

            # grid_sample feature-map
            feat_res4 = F.grid_sample(path_3, point_coords_res4, mode='bilinear', align_corners=True)  # (B, 512, 1, N)
            init_pred = F.grid_sample(out_res8, point_coords_res4, mode='bilinear', align_corners=True)  # (B, 4, 1, N)
            feat_res4 = torch.cat([feat_res4, init_pred], dim=1)  # (B, 512+4, 1, N)

            # prediction (needed to compute loss)
            samples_pred_res4 = self.out_conv_res4(feat_res4[:, :, 0, :])  # (B, 4, N)
            samples_pred_res4 = norm_normalize(samples_pred_res4)  # (B, 4, N) - normalized

            for i in range(B):
                out_res4[i, :, rows_int[i, :], cols_int[i, :]] = samples_pred_res4[i, :, :]

        else:
            # grid_sample feature-map
            feat_map = F.interpolate(path_3, scale_factor=2, mode='bilinear', align_corners=True)
            init_pred = F.interpolate(out_res8, scale_factor=2, mode='bilinear', align_corners=True)
            feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
            B, _, H, W = feat_map.shape

            # try all pixels
            out_res4 = self.out_conv_res4(feat_map.view(B, 256 + 4, -1))  # (B, 4, N)
            out_res4 = norm_normalize(out_res4)  # (B, 4, N) - normalized
            out_res4 = out_res4.view(B, 4, H, W)
            samples_pred_res4 = point_coords_res4 = None
        
        # 1/2 res
        if mode == 'train':
    
            # upsampling ... out_res4: [2, 4, 120, 160] -> out_res4_res2: [2, 4, 240, 320]
            out_res4_res2 = F.interpolate(out_res4, scale_factor=2, mode='bilinear', align_corners=True)
            B, _, H, W = out_res4_res2.shape

            # samples: [B, 1, N, 2]
            point_coords_res2, rows_int, cols_int = sample_points(out_res4_res2.detach(), gt_norm_mask,
                                                                  sampling_ratio=self.sampling_ratio,
                                                                  beta=self.importance_ratio)

            # output (needed for evaluation / visualization)
            out_res2 = out_res4_res2

            # grid_sample feature-map
            feat_res2 = F.grid_sample(path_2, point_coords_res2, mode='bilinear', align_corners=True)  # (B, 256, 1, N)
            init_pred = F.grid_sample(out_res4, point_coords_res2, mode='bilinear', align_corners=True)  # (B, 4, 1, N)
            feat_res2 = torch.cat([feat_res2, init_pred], dim=1)  # (B, 256+4, 1, N)

            # prediction (needed to compute loss)
            samples_pred_res2 = self.out_conv_res2(feat_res2[:, :, 0, :])  # (B, 4, N)
            samples_pred_res2 = norm_normalize(samples_pred_res2)  # (B, 4, N) - normalized

            for i in range(B):
                out_res2[i, :, rows_int[i, :], cols_int[i, :]] = samples_pred_res2[i, :, :]

        else:
            # grid_sample feature-map
            feat_map = F.interpolate(path_2, scale_factor=2, mode='bilinear', align_corners=True)
            init_pred = F.interpolate(out_res4, scale_factor=2, mode='bilinear', align_corners=True)
            feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
            B, _, H, W = feat_map.shape

            out_res2 = self.out_conv_res2(feat_map.view(B, 256 + 4, -1))  # (B, 4, N)
            out_res2 = norm_normalize(out_res2)  # (B, 4, N) - normalized
            out_res2 = out_res2.view(B, 4, H, W)
            samples_pred_res2 = point_coords_res2 = None

        # 1 res
        if mode == 'train':
            # upsampling ... out_res4: [2, 4, 120, 160] -> out_res4_res2: [2, 4, 240, 320]
            out_res2_res1 = F.interpolate(out_res2, scale_factor=2, mode='bilinear', align_corners=True)
            B, _, H, W = out_res2_res1.shape

            # samples: [B, 1, N, 2]
            point_coords_res1, rows_int, cols_int = sample_points(out_res2_res1.detach(), gt_norm_mask,
                                                                  sampling_ratio=self.sampling_ratio,
                                                                  beta=self.importance_ratio)

            # output (needed for evaluation / visualization)
            out_res1 = out_res2_res1

            # grid_sample feature-map
            feat_res1 = F.grid_sample(path_1, point_coords_res1, mode='bilinear', align_corners=True)  # (B, 128, 1, N)
            init_pred = F.grid_sample(out_res2, point_coords_res1, mode='bilinear', align_corners=True)  # (B, 4, 1, N)
            feat_res1 = torch.cat([feat_res1, init_pred], dim=1)  # (B, 128+4, 1, N)

            # prediction (needed to compute loss)
            samples_pred_res1 = self.out_conv_res1(feat_res1[:, :, 0, :])  # (B, 4, N)
            samples_pred_res1 = norm_normalize(samples_pred_res1)  # (B, 4, N) - normalized

            for i in range(B):
                out_res1[i, :, rows_int[i, :], cols_int[i, :]] = samples_pred_res1[i, :, :]

        else:
            # grid_sample feature-map
            feat_map = F.interpolate(path_1, scale_factor=2, mode='bilinear', align_corners=True)
            init_pred = F.interpolate(out_res2, scale_factor=2, mode='bilinear', align_corners=True)
            feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
            B, _, H, W = feat_map.shape

            out_res1 = self.out_conv_res1(feat_map.view(B, 256 + 4, -1))  # (B, 4, N)
            out_res1 = norm_normalize(out_res1)  # (B, 4, N) - normalized
            out_res1 = out_res1.view(B, 4, H, W)
            samples_pred_res1 = point_coords_res1 = None

        return [out_res8, out_res4, out_res2, out_res1], \
               [out_res8, samples_pred_res4, samples_pred_res2, samples_pred_res1], \
               [None, point_coords_res4, point_coords_res2, point_coords_res1]

class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = path_1
        for i, it in enumerate(self.scratch.output_conv):
            out = it(out)
            if i == 4:
                feature = out.clone()
        return [out], [feature]

class Cerberus(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(Cerberus, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet01 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet02 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet03 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet04 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet05 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet06 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet07 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet08 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet09 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet10 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet11 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet12 = _make_fusion_block(features, use_bn)

class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=False), # zyp
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        return super().forward(x).squeeze(dim=1)

class DPTSegmentationModel(DPT):
    def __init__(self, num_classes, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

class DPTNormalModel(DPT_Uncertain):
    def __init__(self, num_classes, path=None, sampling_ratio=None, importance_ratio=None,**kwargs):

        features = kwargs["features"] if "features" in kwargs else 128

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, sampling_ratio, importance_ratio, **kwargs)

        if path is not None:
            self.load(path)

class DPTSegmentationModelMultiHead(DPT):
    def __init__(self, num_classes, output_task_list, path=None, **kwargs):
        self.output_task_list = output_task_list

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = None

        super().__init__(head, **kwargs)
        
        for it in output_task_list:
            setattr(self.scratch, "output_" + it ,nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(features, num_classes, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            ))

        if path is not None:
            self.load(path)

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        outs = list()
        features = list()

        for it in self.output_task_list:
            fun = eval("self.scratch.output_" + it)
            out = path_1
            for j, jt in enumerate(fun):
                out = jt(out)
                if j == 4:
                    feature = out.clone()
            outs.append(out)
            features.append(feature)
        return outs, features

class CerberusSegmentationModelMultiHead(Cerberus):
    def __init__(self, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = None

        super().__init__(head, **kwargs)

        full_output_task_list = ( \
            (2, ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']), \
            (2, ['L','M','R','S','W']), \
            (40, ['Segmentation']) \
        )

        self.full_output_task_list = full_output_task_list
        self.add_module('sigma',nn.Module())

        self.sigma.attribute_sigmas = nn.Parameter(torch.Tensor(1).uniform_(-1.60, 0.0), requires_grad=True)
        self.sigma.affordance_sigmas = nn.Parameter(torch.Tensor(1).uniform_(-1.60, 0.0), requires_grad=True)
        self.sigma.seg_sigmas = nn.Parameter(torch.Tensor(1).uniform_(-1.60, 0.0), requires_grad=True)

        self.sigma.sub_attribute_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[0][1])).uniform_(-1.60, 0.0), requires_grad=True)
        self.sigma.sub_affordance_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[1][1])).uniform_(-1.60, 0.0), requires_grad=True)
        self.sigma.sub_seg_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[2][1])).uniform_(-1.60, 0.0), requires_grad=True)
        
        self.sigma.attribute_sigmas = nn.Parameter(torch.Tensor(1).uniform_(0.20, 1.0), requires_grad=True)
        self.sigma.affordance_sigmas = nn.Parameter(torch.Tensor(1).uniform_(0.20, 1.0), requires_grad=True)
        self.sigma.seg_sigmas = nn.Parameter(torch.Tensor(1).uniform_(0.20, 1.0), requires_grad=True)

        self.sigma.sub_attribute_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[0][1])).uniform_(0.20, 1.0), requires_grad=True)
        self.sigma.sub_affordance_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[1][1])).uniform_(0.20, 1.0), requires_grad=True)
        self.sigma.sub_seg_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[2][1])).uniform_(0.20, 1.0), requires_grad=True)
        


        for (num_classes, output_task_list) in full_output_task_list:
            for it in output_task_list:
                setattr(self.scratch, "output_" + it ,nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(True),
                    nn.Dropout(0.1, False),
                    nn.Conv2d(features, num_classes, kernel_size=1),
                    # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                ))

                setattr(self.scratch, "output_" + it + '_upsample', 
                    Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
                )
            

        if path is not None:
            self.load(path)
        else:
            pass

    def get_attention(self, x ,name):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        x = forward_flex(self.pretrained.model, x, True, name)

        return x

    def forward(self, x ,index):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        if (index == 0):
            path_4 = self.scratch.refinenet04(layer_4_rn)
            path_3 = self.scratch.refinenet03(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet02(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet01(path_2, layer_1_rn)
        elif (index == 1):
            path_4 = self.scratch.refinenet08(layer_4_rn)
            path_3 = self.scratch.refinenet07(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet06(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet05(path_2, layer_1_rn)
        elif(index == 2):
            path_4 = self.scratch.refinenet12(layer_4_rn)
            path_3 = self.scratch.refinenet11(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet10(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet09(path_2, layer_1_rn)
        else:
            assert 0 == 1
        
        output_task_list = self.full_output_task_list[index][1]

        outs = list()

        for it in output_task_list:
            fun = eval("self.scratch.output_" + it)
            out = fun(path_1)
            fun = eval("self.scratch.output_" + it + '_upsample')
            out = fun(out)
            outs.append(out)

        return outs,  [self.sigma.sub_attribute_sigmas, 
                    self.sigma.sub_affordance_sigmas,
                    self.sigma.sub_seg_sigmas, 
                    self.sigma.attribute_sigmas, 
                    self.sigma.affordance_sigmas, 
                    self.sigma.seg_sigmas], []

#################### two head ##################

class TwoHeadDPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(TwoHeadDPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet01 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet02 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet03 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet04 = _make_fusion_block(features, use_bn)

        self.scratch.refinenet05 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet06 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet07 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet08 = _make_fusion_block(features, use_bn)

class GeometryAmphisbaenaModel(TwoHeadDPT):
    def __init__(self, path=None, **kwargs):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = None

        super().__init__(head, **kwargs)

        full_output_task_list = ( \
            (2, ['Wood','Painted','Paper','Glass','Brick','Metal','Flat','Plastic','Textured','Glossy','Shiny']), \
            (2, ['L','M','R','S','W']), \
            (40, ['Segmentation']) \
        )

        self.full_output_task_list = full_output_task_list
        self.add_module('sigma',nn.Module())

        # self.sigma.attribute_sigmas = nn.Parameter(torch.Tensor(1).uniform_(-1.60, 0.0), requires_grad=True)
        # self.sigma.affordance_sigmas = nn.Parameter(torch.Tensor(1).uniform_(-1.60, 0.0), requires_grad=True)
        # self.sigma.seg_sigmas = nn.Parameter(torch.Tensor(1).uniform_(-1.60, 0.0), requires_grad=True)

        # elf.sigma.sub_attribute_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[0][1])).uniform_(-1.60, 0.0), requires_grad=True)
        # self.sigma.sub_affordance_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[1][1])).uniform_(-1.60, 0.0), requires_grad=True)
        # self.sigma.sub_seg_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[2][1])).uniform_(-1.60, 0.0), requires_grad=True)
        
        # self.sigma.attribute_sigmas = nn.Parameter(torch.Tensor(1).uniform_(0.20, 1.0), requires_grad=True)
        # self.sigma.affordance_sigmas = nn.Parameter(torch.Tensor(1).uniform_(0.20, 1.0), requires_grad=True)
        # self.sigma.seg_sigmas = nn.Parameter(torch.Tensor(1).uniform_(0.20, 1.0), requires_grad=True)

        # self.sigma.sub_attribute_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[0][1])).uniform_(0.20, 1.0), requires_grad=True)
        # self.sigma.sub_affordance_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[1][1])).uniform_(0.20, 1.0), requires_grad=True)
        # self.sigma.sub_seg_sigmas = nn.Parameter(torch.Tensor(len(full_output_task_list[2][1])).uniform_(0.20, 1.0), requires_grad=True)
        


        for (num_classes, output_task_list) in full_output_task_list:
            for it in output_task_list:
                setattr(self.scratch, "output_" + it ,nn.Sequential(
                    nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(True),
                    nn.Dropout(0.1, False),
                    nn.Conv2d(features, num_classes, kernel_size=1),
                    # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                ))

                setattr(self.scratch, "output_" + it + '_upsample', 
                    Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
                )
            

        if path is not None:
            self.load(path)
        else:
            pass

    def forward(self, x ,index):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)


        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        if (index == 0):
            path_4 = self.scratch.refinenet04(layer_4_rn)
            path_3 = self.scratch.refinenet03(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet02(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet01(path_2, layer_1_rn)
        elif (index == 2):
            path_4 = self.scratch.refinenet08(layer_4_rn)
            path_3 = self.scratch.refinenet07(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet06(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet05(path_2, layer_1_rn)
        else:
            assert 0 == 1
        
        output_task_list = self.full_output_task_list[index][1]

        outs = list()

        for it in output_task_list:
            fun = eval("self.scratch.output_" + it)
            out = fun(path_1)
            fun = eval("self.scratch.output_" + it + '_upsample')
            out = fun(out)
            outs.append(out)

        return outs,  [#self.sigma.sub_attribute_sigmas, 
                    # self.sigma.sub_affordance_sigmas,
                    # self.sigma.sub_seg_sigmas, 
                    # #self.sigma.attribute_sigmas, 
                    # self.sigma.affordance_sigmas, 
                    # self.sigma.seg_sigmas
                    ], []
