"""Differentiable station delay operator. Inputs/outputs are physical SI/mm.

Approximation: station pressure from MSL and t2m; wet refractivity integrated in
pressure coordinates over 13 levels with a surface node. Not exact GNSS truth.
"""
import torch

LEVELS = (50,100,150,200,250,300,400,500,600,700,850,925,1000)


def profile_delays_mm(t, rh, t2m, msl_pa, height_m, latitude):
    """Last axis=13 levels, top -> bottom. No below-ground integration."""
    t = t.float().clamp(150., 350.)
    rh = rh.float().clamp(0., 100.)
    ts = t2m.float().clamp(180., 340.)
    p = t.new_tensor(LEVELS).expand_as(t)
    ps = (msl_pa.float()/100. * torch.exp(-9.80665*height_m.float()/(287.05*ts))).clamp(51., 1100.)
    above = p < ps[..., None]
    idx = torch.where(above, torch.arange(13, device=t.device), -1).amax(-1).clamp_min(0)
    rs = torch.gather(rh, -1, idx[..., None])[..., 0]
    def vapor(temp, relative):
        return (relative/100.) * 6.112 * torch.exp(17.67*(temp-273.15)/(temp-29.65))
    e = vapor(t, rh)
    es = vapor(ts, rs)
    T = torch.cat((torch.where(above, t, ts[..., None]), ts[..., None]), -1)
    P = torch.cat((torch.minimum(p, ps[..., None]), ps[..., None]), -1)
    E = torch.cat((torch.where(above, e, es[..., None]), es[..., None]), -1).clamp_min(0)
    E = torch.minimum(E, .95*P)
    q = .622*E/(P-.378*E)
    tv = T*(1+.608*q)
    nw = 16.52*E/T + 377600.*E/(T*T)
    integrand = nw*287.05*tv/(9.80665*P)
    dp = (P[..., 1:]-P[..., :-1]).clamp_min(0)
    zwd = 1e-3 * (.5*(integrand[..., 1:]+integrand[..., :-1])*dp).sum(-1)
    denom = 1-.00266*torch.cos(2*torch.deg2rad(latitude.float()))-.00028*(height_m.float()/1000)
    zhd = 2.2768*ps/denom
    return torch.stack((zhd, zwd), -1)


def weather_delays_mm(weather, static, height, latitude, mean, std):
    """Sample station cells FIRST; no global 13-level physics intermediates."""
    b,c,h,w = weather.shape
    if c != 70:
        raise ValueError("physics requires original 70-channel FuXi ordering")
    selected = static[:, 0].reshape(b, -1) > 0
    index = selected.nonzero(as_tuple=False)
    out = weather.new_zeros((b, h*w, 2), dtype=torch.float32)
    if not len(index):
        return out.transpose(1,2).reshape(b,2,h,w)
    bi, cell = index[:,0], index[:,1]
    hs = height.reshape(b,-1)[bi,cell]
    lat = latitude.reshape(b,-1)[bi,cell]
    if not torch.isfinite(hs).all() or not torch.isfinite(lat).all():
        raise ValueError("finite station heights/latitudes required; never substitute zero heights")
    x = weather.float().flatten(2).transpose(1,2)[bi,cell]
    x = x*std.float()[bi] + mean.float()[bi]
    value = profile_delays_mm(x[:,13:26], x[:,52:65], x[:,65], x[:,68], hs, lat)
    out[bi,cell] = value
    return out.transpose(1,2).reshape(b,2,h,w)
