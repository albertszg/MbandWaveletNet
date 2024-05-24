import torch.nn.functional as F
import torch
show_padding = False
if show_padding==True:
    print(torch.__version__)
    a = torch.arange(1, 25).reshape(4, 1, 6).float()
    p1d = (5, 5)  # pad last dim by 5 on each side
    out_constant = F.pad(a, p1d, "constant", 0)  # effectively zero padding
    print('constant', out_constant)

    p1d = (5, 5)  # pad last dim by 5 on each side
    out_reflect = F.pad(a, p1d, "reflect")  # effectively reflect padding
    print('reflect', out_reflect)

    p1d = (5, 5)  # pad last dim by 1 on each side
    out_replicate = F.pad(a, p1d, "replicate")  # effectively replicate padding
    print('replicate', out_replicate)

    p1d = (5, 5)  # pad last dim by 1 on each side
    out_circular = F.pad(a, p1d, "circular")  # effectively circular padding
    print('circular', out_circular)