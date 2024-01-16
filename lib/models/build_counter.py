import importlib

def build_counter(config):

    # Baseline_Counter(config.network,config.dataset.den_factor,config.train.route_size,device)
    module_name = config.network.get("module", None)
    model_name = config.network.get("model", None)

    # 默认名称
    if not module_name:
        module_name = "baseline_counter"
    if not model_name:
        model_name = "Baseline_Counter"

    # 动态导入模块
    module = importlib.import_module(f'lib.models.networks.{module_name}')

    # 获取模型类
    model_class = getattr(module, model_name)

    # 创建模型对象
    model = model_class(config.network,config.dataset.den_factor,config.train.route_size, config.device)

    return model


if __name__ == "__main__":

    import torch
    from mmcv import Config

    config_name = "../../configs/chfloss/SHHA_final.py"
    config = Config.fromfile(config_name)
    config.device=torch.device('cuda:{}'.format(0))
    model = build_counter(config)
    pass
