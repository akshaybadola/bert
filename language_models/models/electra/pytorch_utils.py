import torch
from common_pyutil.system import Semver

torch_version_base = Semver(torch.__version__.split("+")[0])


