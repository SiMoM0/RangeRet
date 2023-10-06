# Vision embedding from (https://github.com/microsoft/torchscale/blob/main/torchscale/component/embedding.py)

import torch
from torch import nn

class VisionEmbedding(nn.Module):
    '''
    Image to Patch Embedding: create patches with a certain number of feature\n
    ```input``` x of shape (B, C, H, W)
    * B = batch size
    * C = number of channels
    * H = image height
    * W = image width
    \n
    output patches of shape (B, P, F)
    * B = batch size
    * P = number of patches
    * F = number of final features
    '''

    def __init__(self, H=64, W=1024, patch_size=4, in_chans=5, embed_dim=128, stride=1):
        super().__init__()
        img_size = (H, W)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride#, padding=(patch_size[0] // 2, patch_size[1] // 2)
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, masked_position=None, **kwargs):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x

# vision patch embedding test
if __name__ == '__main__':
    vembed = VisionEmbedding()
    image = torch.rand(1, 5, 64, 1024)
    print(image.shape)

    out = vembed(image)
    print(out.shape)