#!/usr/bin/env python3
"""从 PyTorch 参考实现导出黄金数据（golden），供 Go 侧逐层对拍。

用法：
    python3 tools/dump_golden.py --upstream upstream --weights ref_weights/weights/trufor.pth.tar \
        --image upstream/test_docker/images/pristine1.jpg --out testdata/golden

产物（全部小端 float32 裸数组 + manifest.json）：
    <out>/dncnn/input.f32              输入张量（PIL 解码后的 RGB/256.0）
    <out>/dncnn/layer_<idx>.f32        DnCNN 每个子模块（conv/bn/relu）的输出
    <out>/weights/<name>.f32          若干权重张量的完整数据，用于校验 Go 权重加载器
    <out>/manifest.json               形状、统计量与环境信息

注意：本脚本只用于生成测试基线，不参与 Go 侧运行。参考环境需钉版本，
官方 README 已说明不同 torch 版本会带来分数级微差。
"""

import argparse
import json
import os
import platform
import sys

import numpy as np


def build_reference(upstream, weights_path):
    here = os.path.dirname(os.path.realpath(__file__))
    shim = os.path.join(here, "timm_shim")
    src = os.path.join(upstream, "test_docker", "src")
    for p in (shim, src):
        if p not in sys.path:
            sys.path.insert(0, p)

    # 上游 config.update_config 固定读取当前工作目录的 trufor.yaml，
    # 这里直接构造配置对象，避免复制/改写上游文件。
    from yacs.config import CfgNode as CN

    cfg = CN()
    cfg.DATASET = CN({"NUM_CLASSES": 2})
    cfg.MODEL = CN()
    cfg.MODEL.NAME = "detconfcmx"
    cfg.MODEL.MODS = ("RGB", "NP++")
    cfg.MODEL.PRETRAINED = ""
    cfg.MODEL.EXTRA = CN(new_allowed=True)
    cfg.MODEL.EXTRA.BACKBONE = "mit_b2"
    cfg.MODEL.EXTRA.DECODER = "MLPDecoder"
    cfg.MODEL.EXTRA.DECODER_EMBED_DIM = 512
    cfg.MODEL.EXTRA.PREPRC = "imagenet"
    cfg.MODEL.EXTRA.BN_EPS = 0.001
    cfg.MODEL.EXTRA.BN_MOMENTUM = 0.1
    cfg.MODEL.EXTRA.DETECTION = "confpool"
    cfg.MODEL.EXTRA.CONF = True
    cfg.freeze()

    import torch
    from models.cmx.builder_np_conf import myEncoderDecoder

    model = myEncoderDecoder(cfg=cfg)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def dump_weights(model, out_dir, names):
    import torch

    sd = model.state_dict()
    os.makedirs(out_dir, exist_ok=True)
    meta = {}
    for name in names:
        if name not in sd:
            raise SystemExit(f"权重中不存在 {name}")
        arr = sd[name].detach().cpu().numpy().astype("<f4")
        path = os.path.join(out_dir, name.replace(".", "_") + ".f32")
        arr.tofile(path)
        meta[name] = {"shape": list(arr.shape), "file": os.path.basename(path)}
    return meta


def stats(a):
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "abs_sum": float(np.abs(a).sum()),
    }


def dump_dncnn(model, image_path, out_dir, crop):
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    if crop:
        cw, ch = crop
        if img.width < cw or img.height < ch:
            raise SystemExit(f"图像太小，无法裁剪 {cw}x{ch}")
        img = img.crop((0, 0, cw, ch))
    rgb = np.array(img)  # HWC uint8，与上游 myDataset 一致
    x = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float) / 256.0  # noqa: E501 (上游用的是 256 不是 255)
    x = x.unsqueeze(0)

    os.makedirs(out_dir, exist_ok=True)
    layers = []

    def save(name, tensor):
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        arr = np.ascontiguousarray(arr.astype("<f4"))
        path = os.path.join(out_dir, f"{name}.f32")
        arr.tofile(path)
        layers.append(
            {
                "name": name,
                "shape": list(arr.shape),
                "file": os.path.basename(path),
                "stats": stats(arr),
            }
        )

    dncnn = model.dncnn
    save("input", x[0])
    with torch.no_grad():
        h = x
        for idx, child in enumerate(dncnn):
            h = child(h)
            save(f"layer_{idx:02d}_{type(child).__name__.lower()}", h)
    return layers, tuple(rgb.shape[:2])


def forward_backbone_traced(model, rgb01, save):
    """按上游 DualSegformer.forward_features 的结构重放，并在检查点导出。

    上游 forward_features 未暴露中间态，这里逐行重写以获得稳定的检查点命名，
    与 Go 侧 Backbone.ForwardTrace 的命名保持一致。
    """
    import torch
    from torch.nn import functional as F

    bb = model.backbone
    modal_x = model.dncnn(rgb01)
    modal_x = torch.tile(modal_x, (3, 1, 1))
    rgb = (rgb01 - torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]) / torch.tensor(
        [0.229, 0.224, 0.225]
    )[None, :, None, None]

    B = rgb.shape[0]
    outs = []
    x_rgb, x_e = rgb, modal_x
    for stage in range(4):
        pe = getattr(bb, f"patch_embed{stage + 1}")
        xpe = getattr(bb, f"extra_patch_embed{stage + 1}")
        x_rgb, H, W = pe(x_rgb)
        x_e, _, _ = xpe(x_e)
        save(f"stage{stage + 1}.rgb_patch_embed", x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2))
        save(f"stage{stage + 1}.np_patch_embed", x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2))
        for j, blk in enumerate(getattr(bb, f"block{stage + 1}")):
            x_rgb = blk(x_rgb, H, W)
            save(f"stage{stage + 1}.block{j}", x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2))
        for j, blk in enumerate(getattr(bb, f"extra_block{stage + 1}")):
            x_e = blk(x_e, H, W)
            save(f"stage{stage + 1}.extra_block{j}", x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2))
        x_rgb = getattr(bb, f"norm{stage + 1}")(x_rgb)
        x_e = getattr(bb, f"extra_norm{stage + 1}")(x_e)
        save(f"stage{stage + 1}.rgb_norm", x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2))
        save(f"stage{stage + 1}.np_norm", x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2))
        x_rgb = x_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_e = x_e.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_rgb, x_e = bb.FRMs[stage](x_rgb, x_e)
        save(f"stage{stage + 1}.frm_rgb", x_rgb)
        save(f"stage{stage + 1}.frm_np", x_e)
        x_fused = bb.FFMs[stage](x_rgb, x_e)
        save(f"stage{stage + 1}.ffm", x_fused)
        outs.append(x_fused)
        # 注意：下一 stage 的输入是 FRM 修正后的特征（x_rgb / x_e），不是 FFM 的融合输出

    _ = F
    return outs, modal_x


def dump_e2e(model, images, out_dir, max_array_pixels=2_000_000, e2e_backbone=False):
    """对给定图像跑完整推理链（等价 trufor_test.py），导出端到端基准。

    map = softmax(logits)[1]，score = sigmoid(det)。像素数超过 max_array_pixels 的图像
    只记录 score 与统计量，避免黄金数据过大。
    """
    import torch
    from PIL import Image
    from torch.nn import functional as F

    os.makedirs(os.path.join(out_dir, "e2e"), exist_ok=True)
    entries = []
    with torch.no_grad():
        for path in images:
            img = Image.open(path).convert("RGB")
            arr = np.array(img)
            x = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0) / 256.0
            captured = {}
            handles = [
                model.decode_head.linear_pred.register_forward_hook(
                    lambda m, i, o: captured.__setitem__("loc4", o[0].detach().numpy().astype("<f4"))
                ),
                model.decode_head_conf.linear_pred.register_forward_hook(
                    lambda m, i, o: captured.__setitem__("conf4", o[0].detach().numpy().astype("<f4"))
                ),
            ]
            out, conf, det, npp = model(x)
            for hd in handles:
                hd.remove()
            logits = out[0].numpy().astype("<f4")
            conf_logits = conf[0].numpy().astype("<f4")
            prob = F.softmax(out, dim=1)[0, 1].numpy().astype("<f4")
            conf_sig = torch.sigmoid(conf)[0, 0].numpy().astype("<f4")
            score = float(torch.sigmoid(det).item())
            name = os.path.basename(path)
            entry = {
                "name": name,
                "format": os.path.splitext(name)[1].lstrip(".").lower(),
                "image_rel": os.path.relpath(path, args.upstream) if False else os.path.join("test_docker", "images", name),
                "input_shape": [3, arr.shape[0], arr.shape[1]],
                "score": score,
                "map_stats": stats(prob),
                "conf_stats": stats(conf_sig),
                "np_stats": stats(npp[0, 0].numpy()),
            }
            if e2e_backbone and entries == []:
                # 为端到端图像额外导出主干检查点（用于把误差定位到具体 stage）
                keep = ["stage1.rgb_patch_embed", "stage1.np_patch_embed", "stage1.block0",
                        "stage1.ffm", "stage2.ffm", "stage3.ffm", "stage4.ffm"]

                def save_bb2(name, tensor):
                    if name not in keep:
                        return
                    f = f"e2e/{os.path.splitext(name)[0] if False else name.replace('.', '_')}.f32"
                    arr2 = np.ascontiguousarray(tensor.detach().cpu().numpy()[0].astype("<f4"))
                    arr2.tofile(os.path.join(out_dir, f))
                    entry.setdefault("backbone_files", {})[name] = {"file": f, "shape": list(arr2.shape)}

                with torch.no_grad():
                    forward_backbone_traced(model, x, save_bb2)

            npx = arr.shape[0] * arr.shape[1]
            if npx <= max_array_pixels:
                def save(tag, arr2):
                    f = f"e2e/{os.path.splitext(name)[0]}_{tag}.f32"
                    np.ascontiguousarray(arr2.astype("<f4")).tofile(os.path.join(out_dir, f))
                    return f

                entry["pixels_file"] = save("pixels", x[0].numpy())
                entry["logits_file"] = save("logits", logits)
                entry["conf_logits_file"] = save("conflogits", conf_logits)
                entry["map_file"] = save("map", prob)
                entry["conf_file"] = save("conf", conf_sig)
                entry["np_file"] = save("np", npp[0, 0].numpy())
                entry["loc4_file"] = save("loc4", captured["loc4"])
                entry["conf4_file"] = save("conf4", captured["conf4"])
            entries.append(entry)
            print(f"  {name}: score={score:.6f}")
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", default="upstream", help="TruFor 源码根目录")
    ap.add_argument("--weights", default="ref_weights/weights/trufor.pth.tar", help="官方权重")
    ap.add_argument("--image", default="upstream/test_docker/images/pristine1.jpg", help="用于生成输入的图像")
    ap.add_argument("--crop", default="96x64", help="裁剪尺寸 WxH（0x0 表示不裁剪）")
    ap.add_argument("--out", default="testdata/golden", help="输出目录")
    ap.add_argument(
        "--e2e",
        default="tampered2.png,pristine1.jpg",
        help="端到端用例（upstream/test_docker/images 下的文件名，逗号分隔）",
    )
    ap.add_argument("--e2e-backbone", action="store_true", help="为端到端用例导出主干检查点（排错用）")
    ap.add_argument("--backbone-size", default="64x64", help="backbone 快照的输入尺寸 WxH（0x0 表示跳过）")
    args = ap.parse_args()

    import torch

    cw, ch = 0, 0
    if args.crop and args.crop.lower() != "0x0":
        cw, ch = (int(v) for v in args.crop.lower().split("x"))

    model, cfg = build_reference(args.upstream, args.weights)

    weight_names = [
        "dncnn.0.weight",
        "dncnn.0.bias",
        "dncnn.2.weight",
        "dncnn.3.weight",
        "dncnn.3.bias",
        "dncnn.3.running_mean",
        "dncnn.3.running_var",
        "dncnn.47.weight",
        "dncnn.47.bias",
        "backbone.patch_embed1.proj.weight",
    ]
    wmeta = dump_weights(model, os.path.join(args.out, "weights"), weight_names)
    layers, img_hw = dump_dncnn(model, args.image, os.path.join(args.out, "dncnn"), (cw, ch))

    backbone_layers = []
    if args.backbone_size and args.backbone_size.lower() != "0x0":
        bw, bh = (int(v) for v in args.backbone_size.lower().split("x"))
        bb_dir = os.path.join(args.out, "backbone")

        def save_bb(name, tensor):
            arr = np.ascontiguousarray(tensor.detach().cpu().numpy()[0].astype("<f4"))
            path = os.path.join(bb_dir, name.replace(".", "_") + ".f32")
            arr.tofile(path)
            backbone_layers.append(
                {
                    "name": name,
                    "shape": list(arr.shape),
                    "file": os.path.relpath(path, args.out),
                    "stats": stats(arr),
                }
            )

        os.makedirs(bb_dir, exist_ok=True)
        gen = torch.Generator().manual_seed(20240501)
        rgb01 = torch.rand((1, 3, bh, bw), generator=gen)
        save_bb("input.rgb01", rgb01)
        with torch.no_grad():
            outs, modal = forward_backbone_traced(model, rgb01, save_bb)
            loc4 = model.decode_head(outs)
            conf4 = model.decode_head_conf(outs)
        save_bb("logits.loc4", loc4)
        save_bb("logits.conf4", conf4)
        save_bb("input.modal_x", modal)

    e2e_entries = []
    if args.e2e:
        names = [n.strip() for n in args.e2e.split(",") if n.strip()]
        images = [os.path.join(args.upstream, "test_docker", "images", n) for n in names]
        images = [p for p in images if os.path.isfile(p)]
        if images:
            e2e_entries = dump_e2e(model, images, args.out, e2e_backbone=args.e2e_backbone)

    bn_eps = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            bn_eps[name] = mod.eps
        elif isinstance(mod, torch.nn.LayerNorm):
            bn_eps[name] = mod.eps

    manifest = {
        "env": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "image": os.path.basename(args.image),
            "image_crop_wh": [cw, ch],
            "image_hw": list(img_hw),
        },
        "config": {
            "backbone": cfg.MODEL.EXTRA.BACKBONE,
            "decoder": cfg.MODEL.EXTRA.DECODER,
            "embed_dim": cfg.MODEL.EXTRA.DECODER_EMBED_DIM,
            "bn_eps": cfg.MODEL.EXTRA.BN_EPS,
        },
        "norm_eps": bn_eps,
        "weights": wmeta,
        "dncnn_layers": layers,
        "backbone_layers": backbone_layers,
        "e2e": e2e_entries,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(
        f"已写入 {args.out}: {len(manifest['dncnn_layers'])} 个 DnCNN 层快照, "
        f"{len(manifest['backbone_layers'])} 个 backbone 快照, "
        f"{len(manifest['e2e'])} 个端到端用例, {len(wmeta)} 个权重校验张量"
    )


if __name__ == "__main__":
    main()
