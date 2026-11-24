from tensorion.config_resolvers import register_resolvers

from . import optimizers

register_resolvers()

__all__ = ["optimizers"]
