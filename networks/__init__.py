from networks.FPTrans import FPTrans
from networks.FPTrans_fixed import FPTransFixed


__networks = {
    'fptrans': FPTrans,
    'fptrans_fixed': FPTransFixed
}


def load_model(opt, logger, *args, **kwargs):
    if opt.network.lower() in __networks:
        model = __networks[opt.network.lower()](opt, logger, *args, **kwargs)
        if opt.print_model:
            print(model)
        return model
    else:
        raise ValueError(f'Not supported network: {opt.network}. {list(__networks.keys())}')
