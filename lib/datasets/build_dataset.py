import lib.datasets as datasets

def build_dataset(config):
    if config.dataset.name == "SHHA_Chf":
        return build_dataset_shha_chf(config)

    if config.dataset.name == "SHHA_Sim_Match":
        return build_dataset_shha_chf(config)

    # default
    train_dataset = eval('datasets.' + config.dataset.name)(
            root=config.dataset.root,
            list_path=config.dataset.train_set,
            num_samples=None,
            num_classes=config.dataset.num_classes,
            multi_scale=config.train.multi_scale,
            flip=config.train.flip,
            ignore_label=None,
            base_size=config.train.base_size,
            crop_size=config.train.image_size,
            min_unit=config.train.route_size,
            scale_factor=config.train.scale_factor)

    test_dataset = eval('datasets.' + config.dataset.name)(
        root=config.dataset.root,
        list_path=config.dataset.test_set,
        num_samples=None,
        num_classes=config.dataset.num_classes,
        multi_scale=False,
        flip=False,
        base_size=config.test.base_size,
        crop_size=(None, None),
        min_unit=config.train.route_size,
        downsample_rate=1)

    return train_dataset, test_dataset


def build_dataset_shha_chf(config):
    train_dataset = eval('datasets.' + config.dataset.name)(
        config=config,
        root=config.dataset.root,
        list_path=config.dataset.train_set,
        num_samples=None,
        num_classes=config.dataset.num_classes,
        multi_scale=config.train.multi_scale,
        flip=config.train.flip,
        ignore_label=None,
        base_size=config.train.base_size,
        crop_size=config.train.image_size,
        min_unit=config.train.route_size,
        scale_factor=config.train.scale_factor)

    test_dataset = eval('datasets.' + config.dataset.name)(
        config=config,
        root=config.dataset.root,
        list_path=config.dataset.test_set,
        num_samples=None,
        num_classes=config.dataset.num_classes,
        multi_scale=False,
        flip=False,
        base_size=config.test.base_size,
        crop_size=(None, None),
        min_unit=config.train.route_size,
        downsample_rate=1)
    return train_dataset, test_dataset