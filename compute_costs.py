# compute_costs.py

import argparse
import time
import torch
import torch.nn as nn

from models import MLPClassifier, BPANClassifier, ConvBackboneCIFAR, ConvWrapper


def count_linear_flops(m, inp, out):
    x = inp[0]  # [B, in_features]
    B = x.shape[0]
    in_f = m.in_features
    out_f = m.out_features
    return 2 * B * in_f * out_f


def count_conv2d_flops(m, inp, out):
    x = inp[0]  # [B, Cin, H, W]
    B, Cin, H, W = x.shape
    Cout = m.out_channels
    KH, KW = m.kernel_size
    H_out, W_out = out.shape[2], out.shape[3]
    return 2 * B * Cin * Cout * KH * KW * H_out * W_out


def add_flop_hooks(model):
    flops = {"total": 0}

    def hook_fn(module, inp, out):
        if isinstance(module, nn.Linear):
            flops["total"] += count_linear_flops(module, inp, out)
        elif isinstance(module, nn.Conv2d):
            flops["total"] += count_conv2d_flops(module, inp, out)

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.register_forward_hook(hook_fn)
    return flops


def measure(model, input_shape, device, n_warmup=10, n_runs=50):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dummy = torch.randn(*input_shape, device=device)

    flops_dict = add_flop_hooks(model)

    # warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.time()

    avg_latency_ms = (t1 - t0) * 1000.0 / n_runs
    total_flops = flops_dict["total"] / n_runs

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mem = 0.0

    return avg_latency_ms, total_flops, peak_mem


def main():
    parser = argparse.ArgumentParser(
        description="Compute approximate FLOPs, latency and memory for heads."
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "cifar10"])
    parser.add_argument("--model", type=str, default="mlp_head",
                        choices=["mlp_head", "bpan_head", "conv_mlp", "conv_bpan"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--bpan_T", type=int, default=6)
    parser.add_argument("--ei_ratio", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.dataset in ["mnist", "fashion_mnist"]:
        input_dim = 28 * 28
        input_shape = (64, input_dim)
        n_classes = 10
    elif args.dataset == "cifar10":
        input_shape = (64, 3, 32, 32)
        n_classes = 10
    else:
        raise ValueError

    if args.model == "mlp_head":
        model = MLPClassifier(
            input_dim=input_shape[1],
            n_classes=n_classes,
            hidden=args.hidden
        )
    elif args.model == "bpan_head":
        if args.ei_ratio is not None:
            r = args.ei_ratio
            n_exc = int(round(args.hidden * r / (1.0 + r)))
            n_exc = max(1, min(n_exc, args.hidden - 1))
            n_inh = args.hidden - n_exc
        else:
            n_exc = args.hidden // 2
            n_inh = max(4, args.hidden // 8)
        model = BPANClassifier(
            input_dim=input_shape[1],
            n_classes=n_classes,
            n_exc=n_exc,
            n_inh=n_inh,
            T=args.bpan_T
        )
    elif args.model in ["conv_mlp", "conv_bpan"]:
        backbone = ConvBackboneCIFAR()
        if args.model == "conv_mlp":
            head = MLPClassifier(
                input_dim=backbone.out_dim,
                n_classes=n_classes,
                hidden=args.hidden
            )
        else:
            if args.ei_ratio is not None:
                r = args.ei_ratio
                n_exc = int(round(args.hidden * r / (1.0 + r)))
                n_exc = max(1, min(n_exc, args.hidden - 1))
                n_inh = args.hidden - n_exc
            else:
                n_exc = args.hidden // 2
                n_inh = max(4, args.hidden // 8)
            head = BPANClassifier(
                input_dim=backbone.out_dim,
                n_classes=n_classes,
                n_exc=n_exc,
                n_inh=n_inh,
                T=args.bpan_T
            )
        model = ConvWrapper(backbone, head)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    latency_ms, flops, peak_mem = measure(model, input_shape, args.device)

    print(f"Model: {args.model} on {args.dataset}, hidden={args.hidden}")
    print(f"Trainable params: {n_params}")
    print(f"Approx FLOPs / forward (batch {input_shape[0]}): {flops:.2e}")
    print(f"Avg latency: {latency_ms:.3f} ms")
    if peak_mem > 0:
        print(f"Peak CUDA memory: {peak_mem:.2f} MiB")


if __name__ == "__main__":
    main()
