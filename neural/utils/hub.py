from torch.hub import load_state_dict_from_url
from copy import deepcopy

__all__ = ['configure_model']


def configure_model(configurations):
    def wrapper(model_fn):
        def create_fn(*args, config=None, pretrained=True, ** kwargs):
            if config is not None:
                try:
                    config = deepcopy(configurations[config])
                except KeyError:
                    raise ValueError('configuration does not exist: {}'
                                     .format(config))
                state_dict_url = config.pop('state_dict')
                kwargs.update(config)
                model = model_fn(*args, **kwargs)
                if pretrained:
                    state_dict = load_state_dict_from_url(
                        state_dict_url,
                        map_location='cpu',
                        check_hash=True)
                    model.load_state_dict(state_dict)
                return model
            else:
                return model_fn(*args, **kwargs)

        return create_fn

    return wrapper
