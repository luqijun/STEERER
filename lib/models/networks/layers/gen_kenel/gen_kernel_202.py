from .untils import *
from .gen_kernel_201 import GenerateKernelLayer201

class GenerateKernelLayer202(GenerateKernelLayer201):
    def __init__(self, config):
        hidden_channels = config.gen_kernel.get("hidden_channels", 256)
        super(GenerateKernelLayer202, self).__init__(config)
        # 初级的提取
        self.dilation_num = 1
        kernel_blocks, extra_conv_kernels = self.make_kernel_extractor(config, hidden_channels,
                                                                  dilation_num=self.dilation_num)
        self.kernel_blocks = nn.Sequential(*kernel_blocks)
        self.extra_conv_kernels = nn.Sequential(*extra_conv_kernels)
        pass

    def make_kernel_extractor(self, config, hidden_channels, dilation_num=3):

        # extract kernerls
        avg_pool =  nn.AvgPool2d(2, 2)
        conv1 = make_conv_layer(hidden_channels, 3, 1, 1)  # 3*3

        kernel_blocks = [avg_pool, conv1]

        # 创建不同的dialation kernel
        extra_conv_group = []
        for i in range(dilation_num - 1):
            cov_list = nn.Sequential(*[make_conv_layer(hidden_channels, 3, 1, p1=1, p2=1) for i, c in
                                       enumerate(config.head.stages_channel)])
            extra_conv_group.append(cov_list)

        return kernel_blocks, extra_conv_group