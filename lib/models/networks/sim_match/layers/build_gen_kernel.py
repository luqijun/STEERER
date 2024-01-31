from .gen_kenel import *

def build_gen_kernel(config):
    gen_kernel_name = config.gen_kernel.name
    return eval(gen_kernel_name)(config)