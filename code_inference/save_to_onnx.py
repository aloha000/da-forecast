import sys
import xarray as xr
import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--fcst_model_dir', type=str, default='../model')
parser.add_argument('--input_dir', type=str, default='../test_data')
parser.add_argument('--output_dir', type=str, default='../output')
parser.add_argument('--date_str', type=str, default='2023070100')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dtype', type=str, default='f32')
parser.add_argument('--fcst_steps', type=int, default=2)
args = parser.parse_args()
import warnings
warnings.simplefilter("ignore")


def get_date(start_str, end_str, step):
    start_time = pd.to_datetime(start_str, format="%Y%m%d%H")
    end_time = pd.to_datetime(end_str, format="%Y%m%d%H")
    dates = []
    for date in pd.date_range(start_time, end_time, freq=f"{step}h"):
        dates.append(date)
    return dates


def chunk_time(ds):
    dims = {k: v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds


def read_analysis(data_dir, date):
    date_str1 = (date - pd.Timedelta(hours=6)).strftime('%Y%m%d%H')
    date_str2 = date.strftime('%Y%m%d%H')
    data1 = xr.open_dataset(f"{data_dir}/{date_str1}.nc").z.values
    data2 = xr.open_dataset(f"{data_dir}/{date_str2}.nc").z.values
    data = np.concatenate([data1, data2], axis=0)
    return data


def time_encoding(init_time, num_steps, freq=6):
    init_time = np.array([init_time])
    tembs = []
    for i in range(num_steps):
        hours = np.array([pd.Timedelta(hours=t * freq) for t in [i - 1, i, i + 1]])
        times = init_time[:, None] + hours[None]
        times = [pd.Period(t, 'H') for t in times.reshape(-1)]
        times = [(p.day_of_year / 366, p.hour / 24) for p in times]
        temb = torch.from_numpy(np.array(times, dtype=np.float32))
        temb = torch.cat([temb.sin(), temb.cos()], dim=-1)
        temb = temb.reshape(1, -1)
        tembs.append(temb)
    return torch.stack(tembs)


def save_nc(outputs, date):
    if outputs.device.type == 'cuda':
        outputs = outputs.cpu().numpy()
    else:
        outputs = outputs.numpy()
    date_str = date.strftime('%Y%m%d%H')
    channel = ['z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500',
               'z600', 'z700', 'z850', 'z925', 'z1000', 't50', 't100', 't150',
               't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850',
               't925', 't1000', 'u50', 'u100', 'u150', 'u200', 'u250', 'u300',
               'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000', 'v50',
               'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600',
               'v700', 'v850', 'v925', 'v1000', 'r50', 'r100', 'r150', 'r200',
               'r250', 'r300', 'r400', 'r500', 'r600', 'r700', 'r850', 'r925',
               'r1000', 't2m', 'u10', 'v10', 'msl', 'tp']
    lat = np.load(f'{args.input_dir}/latitude_era5.npy')[:, 0]
    lon = np.load(f'{args.input_dir}/longitude_era5.npy')[0, :]
    ds = xr.Dataset({'z': (['time', 'step', 'channel', 'lat', 'lon'], outputs.astype(np.float32))},
                    coords={'time': [date],
                            'step': np.arange(1, args.fcst_steps + 1).astype(np.int32) * 6,
                            'channel': channel,
                            'lat': lat,
                            'lon': lon})
    ds = chunk_time(ds)
    ds['time'].encoding['dtype'] = 'float64'
    ds['channel'] = ds['channel'].astype('<U5')
    os.makedirs(f"{args.output_dir}", exist_ok=True)
    comp = dict(zlib=True, complevel=4)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(f"{args.output_dir}/{date_str}.nc", encoding=encoding)
    return None


def load_model(model_path, model):
    print(f'load from {model_path} ...')
    model_state = {}
    chkpt = torch.load(model_path, map_location=torch.device("cpu"))
    for k, v in chkpt["model"].items():
        k = k.replace('decoder.', '')
        model_state[k] = v
    model.load_state_dict(model_state, strict=False)
    return model


def inference(model, inputs, num_steps):
    input, tembs, const = inputs
    B, _, C, H, W = input.shape
    outputs = input.new_zeros(1, num_steps, *input.shape[-3:])
    for i in range(num_steps):
        output = model(input, temb=tembs[i], const=const)
        output = output.reshape(B, 1, C, H, W)
        output = output + input[:, -1:]
        print(f'step: {i + 1}')
        outputs[:, i] = output[0]
        input = torch.cat([input, output], dim=1)[:, -2:]
    return outputs


def to_onnx(model, inputs):
    input, tembs, const = inputs   
    model.eval()
    import pdb; pdb.set_trace()
    # model(input, tembs[0].half(), const.half())
    model(input, tembs[0], const)
    torch.onnx.export(
        # model.half(),
        model,
        (input, tembs[0], const),
        # (input, tembs[0].half(), const.half()),
        f"{args.fcst_model_dir}/onnx/model.onnx",
        input_names=["input", "temb", "const"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "temb": {0: "batch_size"},
            "const": {0: "batch_size"}
        },
        opset_version=13,
        do_constant_folding=False
    )
    return None


def main():
    sys.path.append(args.fcst_model_dir)
    from fuxi import UTransformer
    model = UTransformer(in_chans=75,
                         out_chans=70,
                         in_frames=2,
                         image_size=(720, 1440),
                         window_size=9,
                         patch_size=4,
                         down_times=1,
                         embed_dim=1536,
                         num_heads=24,
                         depths=[12, 12, 12, 12])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == 'f16' else torch.float32
    
    model = model.to(dtype=dtype, device=device)
    model = load_model(f'{args.fcst_model_dir}/fuxi_short.pth', model)
    
    conds = np.load(f"{args.fcst_model_dir}/conds.npy")
    conds = torch.from_numpy(conds).to(dtype=dtype, device=device)
    date = pd.to_datetime(args.date_str, format="%Y%m%d%H")
    with torch.no_grad():
        fuxi_input_data = read_analysis(args.input_dir, date)
        fuxi_input_data = torch.from_numpy(fuxi_input_data).to(device, dtype).unsqueeze(0)
        fuxi_input_tmbs = time_encoding(date, args.fcst_steps).to(device, dtype)
        
        inputs = (fuxi_input_data, fuxi_input_tmbs, conds)
        to_onnx(model, inputs)


if __name__ == '__main__':
    main()
