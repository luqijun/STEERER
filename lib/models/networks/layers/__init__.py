from .attention import *
from .dsnet import *
from .gaussian import *
from lib.models.networks.layers.gen_kenel.gen_kernel import *
from .nested_tensor import *
from .position_encoding import *
from .segmentation import *
from .trans_decoder_wrapper import *
from .transformer import *
from .transition import *
from .twoway_transformer import *
from .build_gen_kernel import *
from .matcher import *
from .mask2former_head import *
from .mask2former_head1 import *
from .mask2former_head2 import *
from .mask2former_head3 import *
from .mask2former_head3_1 import *
from .mask2former_head3_2 import *
from .mask2former_head3_3 import *
from .mask2former_head3_3_1 import *
from .mask2former_head3_4 import *
from .mask2former_head4 import *
from .mask2former_head4_1 import *
from .mask2former_head4_2 import *
from .mask2former_head4_3 import *
from .mask2former_head4_4 import *
from .mask2former_head_v2_1 import *
from .mask2former_head_v2_2 import *
from .mask2former_head_v2_3 import *
from .mask2former_head_v3_1 import *
from .mask2former_head_v3_2 import *
from .mask2former_head_v3_2_1 import *
from .mask2former_head_v3_2_2 import *
from .mask2former_head_v3_2_3 import *
from .mask2former_head_v3_2_4 import *
from .mask2former_head_v3_2_5 import *
from .mask2former_head_v3_2_5_1 import *
from .mask2former_head_v3_2_5_2 import *
from .mask2former_head_v3_2_6 import *
from .anchor_points import *

# import pkgutil
# import inspect
# import importlib
#
# # 遍历当前目录下的所有模块
# for _, module_name, _ in pkgutil.iter_modules(__path__):
#     # 动态导入模块
#     module = importlib.import_module('.' + module_name, __name__)
#
#     # 遍历模块中的所有类
#     for name, obj in inspect.getmembers(module):
#         if inspect.isclass(obj):
#             # 动态导入类
#             globals()[name] = obj

def freeze_model(model):
    for (name, param) in model.named_parameters():
        param.requires_grad = False
