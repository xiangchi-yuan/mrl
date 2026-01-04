import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def ensure_tokenizer_downloaded(model_path, cache_dir="./cache"):
    print(f"[Tokenizer] Downloading / loading tokenizer for: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
 
    if os.path.isdir(model_path):
        tokenizer.save_pretrained(model_path)
        print(f"[Tokenizer] Saved tokenizer files into local dir: {model_path}")
    else:
        print(f"[Tokenizer] Tokenizer cached under: {cache_dir}")

    return tokenizer

def load_state_dict(model_path, device="cpu"):
    ensure_tokenizer_downloaded(model_path, cache_dir="./cache")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    sd = {k: v.detach().to(device) for k, v in model.state_dict().items()}
    del model
    return sd

def compute_overlap_stats_elementwise_n(base_sd, task_sds, threshold=1e-5):
    n = len(task_sds)
    assert n >= 1, "task_sds 至少需要一个任务模型"

    common_keys = set(base_sd.keys())
    for sd in task_sds:
        common_keys &= set(sd.keys())

    counters = []
    for _ in range(n):
        counters.append({
            "total": 0,                           
            "total_elems": 0,                   
            "overlap_counts": [0 for _ in range(n)],
        })

    for name in common_keys:
        base = base_sd[name]
        tensors = [sd[name] for sd in task_sds]

        m_list = [(t - base).abs() > threshold for t in tensors]
        sum_mask = torch.stack(m_list, dim=0).sum(dim=0)
        num_elems_param = base.numel()

        for i in range(n):
            m_i = m_list[i]
            counters[i]["total_elems"] += num_elems_param

            num_changed_i = m_i.sum().item()
            counters[i]["total"] += num_changed_i

            if num_changed_i == 0:
                continue

            for k in range(n):
                target_total_models = k + 1
                mask_k = m_i & (sum_mask == target_total_models)
                counters[i]["overlap_counts"][k] += mask_k.sum().item()

    results = []
    for i in range(n):
        total = counters[i]["total"]
        total_elems = counters[i]["total_elems"]
        overlap_counts = counters[i]["overlap_counts"]

        if total == 0:
            overlap_pcts = [0.0 for _ in range(n)]
        else:
            overlap_pcts = [c / total * 100.0 for c in overlap_counts]

        results.append({
            "total_changed_elems": total,
            "total_elems": total_elems,
            "overlap_counts": overlap_counts,
            "overlap_pcts": overlap_pcts,
        })

    return results

def compute_task_vectors(base_sd, task_sds):
    task_vecs = []
    common_keys = set(base_sd.keys())
    for sd in task_sds:
        common_keys &= set(sd.keys())

    for sd in task_sds:
        tv = {}
        for k in common_keys:
            tv[k] = sd[k] - base_sd[k]
        task_vecs.append(tv)

    return task_vecs

def agentic_reinforcement_merge_extract_unique(base_sd, task_vecs, target_index, threshold=1e-5):
    n = len(task_vecs)
    assert n >= 1, "需要至少一个任务向量"
    if target_index < 0 or target_index >= n:
        raise ValueError(f"target_index {target_index} 超出范围 (共有 {n} 个模型)")

    merged_sd = {}

    common_keys = set(base_sd.keys())
    for tv in task_vecs:
        common_keys &= set(tv.keys())
    
    print(f"[ExtractUnique] Extracting unique parameters from model index: {target_index} ...")

    for name in common_keys:
        diffs = torch.stack([tv[name] for tv in task_vecs], dim=0)
        change_mask = (diffs.abs() > threshold)
        change_mask_f = change_mask.float()
        count = change_mask_f.sum(dim=0)

        is_unique_spot = (count == 1)
        is_target_active = change_mask[target_index]
        final_mask = is_unique_spot & is_target_active
        
        target_diff = diffs[target_index]
        diff_final = torch.where(final_mask, target_diff, torch.zeros_like(target_diff))

        merged_sd[name] = base_sd[name] + diff_final/3

    for k in base_sd.keys():
        if k not in merged_sd:
            merged_sd[k] = base_sd[k].clone()

    return merged_sd

def agentic_reinforcement_merge_ties(base_sd, task_vecs, threshold=1e-5):
    n = len(task_vecs)
    assert n >= 1, "需要至少一个任务向量"

    param_value_mask_rate = 0.9
    scaling_coefficient = 1.0

    merged_sd = {}

    common_keys = set(base_sd.keys())
    for tv in task_vecs:
        common_keys &= set(tv.keys())

    for name in common_keys:
        diffs = torch.stack([tv[name] for tv in task_vecs], dim=0)
        change_mask = (diffs.abs() > threshold)
        change_mask_f = change_mask.float()
        count = change_mask_f.sum(dim=0)
        overlap_mask = (count >= 2)
        non_overlap_mask = (count == 1)

        sum_diff = (diffs * change_mask_f).sum(dim=0)
        denom = torch.clamp(count, min=1.0)
        avg_diff = sum_diff / denom
        diff_final = torch.where(count > 0, avg_diff, torch.zeros_like(avg_diff))

        if overlap_mask.any():
            diffs_flat = diffs.reshape(n, -1)
            change_mask_flat = change_mask.reshape(n, -1)
            overlap_flat = overlap_mask.reshape(-1)

            overlap_indices = overlap_flat.nonzero(as_tuple=False).squeeze(-1)
            if overlap_indices.numel() > 0:
                overlap_diffs = diffs_flat[:, overlap_indices]
                overlap_change = change_mask_flat[:, overlap_indices].float()
                overlap_diffs = overlap_diffs * overlap_change

                def prune_task_vectors(flattened_param: torch.Tensor,
                                       mask_rate: float) -> torch.Tensor:
                    if mask_rate <= 0.0:
                        return flattened_param

                    num_models, d = flattened_param.shape
                    num_mask = int(d * mask_rate)
                    if num_mask <= 0:
                        return flattened_param

                    abs_val = flattened_param.abs()
                    kth_values, _ = abs_val.kthvalue(k=num_mask, dim=1, keepdim=True)
                    mask = abs_val >= kth_values
                    pruned = flattened_param * mask
                    return pruned

                pruned = prune_task_vectors(overlap_diffs, param_value_mask_rate)

                if (pruned != 0).sum() > 0:
                    pos_mass = torch.clamp(pruned, min=0).sum(dim=0)
                    neg_mass = torch.clamp(-pruned, min=0).sum(dim=0)

                    sign_diff = pos_mass - neg_mass
                    gamma_m = torch.sign(sign_diff)

                    gamma_row = gamma_m.unsqueeze(0)

                    preserve_mask = (
                        ((gamma_row > 0) & (pruned > 0)) |
                        ((gamma_row < 0) & (pruned < 0))
                    )

                    preserved = pruned * preserve_mask

                    counts = (preserved != 0).sum(dim=0).float()
                    sum_preserved = preserved.sum(dim=0)

                    merged_overlap = sum_preserved / torch.clamp(counts, min=1.0)
                    merged_overlap = scaling_coefficient * merged_overlap

                    diff_final_flat = diff_final.reshape(-1)
                    diff_final_flat[overlap_indices] = merged_overlap
                    diff_final = diff_final_flat.view_as(diff_final)

        merged_sd[name] = base_sd[name] + diff_final

    for k in base_sd.keys():
        if k not in merged_sd:
            merged_sd[k] = base_sd[k].clone()

    return merged_sd

def agentic_reinforcement_merge(base_sd, task_vecs, threshold=1e-5):
    n = len(task_vecs)
    assert n >= 1, "需要至少一个任务向量"

    merged_sd = {}

    common_keys = set(base_sd.keys())
    for tv in task_vecs:
        common_keys &= set(tv.keys())

    for k in common_keys:
        diffs = torch.stack([tv[k] for tv in task_vecs], dim=0)
        change_mask = (diffs.abs() > threshold)
        change_mask_f = change_mask.float()

        sum_diff = (diffs * change_mask_f).sum(dim=0)
        count = change_mask_f.sum(dim=0)

        denom = torch.clamp(count, min=1.0)

        avg_diff = sum_diff / denom
        diff_final = torch.where(count > 0, avg_diff, torch.zeros_like(sum_diff))

        merged_sd[k] = base_sd[k] + diff_final

    for k in base_sd.keys():
        if k not in merged_sd:
            merged_sd[k] = base_sd[k].clone()

    return merged_sd

def agentic_reinforcement_merge_rescale(base_sd, task_vecs, threshold=1e-5, r=1.0):
    n = len(task_vecs)
    assert n >= 1, "需要至少一个任务向量"

    r = float(r)
    if r <= 1.0:
        r = 1.0

    common_keys = set(base_sd.keys())
    for tv in task_vecs:
        common_keys &= set(tv.keys())
    common_keys = sorted(list(common_keys))

    if not common_keys:
        return {k: v.clone() for k, v in base_sd.items()}

    changed_counts = [0 for _ in range(n)]
    overlap_counts = [0 for _ in range(n)]

    for name in common_keys:
        diffs = torch.stack([tv[name] for tv in task_vecs], dim=0)
        change_mask = (diffs.abs() > threshold)
        sum_change = change_mask.sum(dim=0)

        overlap_any = (sum_change >= 2)

        change_flat = change_mask.view(n, -1)
        overlap_flat = overlap_any.view(-1)

        for j in range(n):
            cj = change_flat[j]
            changed_counts[j] += cj.sum().item()
            overlap_counts[j] += (cj & overlap_flat).sum().item()

    overlap_ratios = []
    rescales = []
    for j in range(n):
        if changed_counts[j] == 0:
            ratio = 0.0
        else:
            ratio = overlap_counts[j] / changed_counts[j]
        overlap_ratios.append(ratio)

        rescale_j = 1.0 + (r - 1.0) * max(0.0, min(1.0, float(ratio)))
        rescales.append(rescale_j)

    print("[OverlapAware] overlap_ratios per task:", overlap_ratios)
    print("[OverlapAware] rescale per task:", rescales)

    merged_sd = {}

    for name in common_keys:
        diffs = torch.stack([tv[name] for tv in task_vecs], dim=0)

        change_mask = (diffs.abs() > threshold)
        change_mask_f = change_mask.float()

        sum_diff = (diffs * change_mask_f).sum(dim=0)
        count = change_mask_f.sum(dim=0)

        denom = torch.clamp(count, min=1.0)

        avg_diff = sum_diff / denom

        zero = torch.zeros_like(sum_diff)
        non_overlap_mask = (count == 1)
        overlap_mask = (count >= 2)

        rescales_tensor = torch.tensor(
            rescales,
            dtype=diffs.dtype,
            device=diffs.device,
        ).view((n,) + (1,) * (diffs.dim() - 1))

        weighted_sum = (diffs * change_mask_f * rescales_tensor).sum(dim=0)

        diff_final = zero
        diff_final = torch.where(overlap_mask, avg_diff, diff_final)
        diff_final = torch.where(non_overlap_mask, weighted_sum, diff_final)

        merged_sd[name] = base_sd[name] + diff_final

    for k in base_sd.keys():
        if k not in merged_sd:
            merged_sd[k] = base_sd[k].clone()

    return merged_sd

def agentic_reinforcement_merge_rescale_v2(base_sd, task_vecs, threshold=1e-5, r=1.1):
    n = len(task_vecs)
    assert n >= 1, "需要至少一个任务向量"

    r = float(r)
    if r <= 1.0:
        r = 1.0

    common_keys = set(base_sd.keys())
    for tv in task_vecs:
        common_keys &= set(tv.keys())
    common_keys = sorted(list(common_keys))

    if not common_keys:
        return {k: v.clone() for k, v in base_sd.items()}

    changed_counts = [0 for _ in range(n)]
    overlap_counts = [0 for _ in range(n)]

    for name in common_keys:
        diffs = torch.stack([tv[name] for tv in task_vecs], dim=0)
        change_mask = (diffs.abs() > threshold)
        sum_change = change_mask.sum(dim=0)

        overlap_any = (sum_change >= 2)

        change_flat = change_mask.view(n, -1)
        overlap_flat = overlap_any.view(-1)

        for j in range(n):
            cj = change_flat[j]
            changed_counts[j] += cj.sum().item()
            overlap_counts[j] += (cj & overlap_flat).sum().item()

    overlap_ratios = []
    rescales = []
    for j in range(n):
        if changed_counts[j] == 0:
            ratio = 0.0
        else:
            ratio = overlap_counts[j] / (changed_counts[j]-overlap_counts[j])
        overlap_ratios.append(ratio)

        rescale_j = 1.0 + (r - 1.0) * min(2, min(1.0, float(ratio)))
        rescales.append(rescale_j)

    print("[OverlapAware] overlap_ratios per task:", overlap_ratios)
    print("[OverlapAware] rescale per task:", rescales)

    merged_sd = {}

    for name in common_keys:
        diffs = torch.stack([tv[name] for tv in task_vecs], dim=0)

        change_mask = (diffs.abs() > threshold)
        change_mask_f = change_mask.float()

        sum_diff = (diffs * change_mask_f).sum(dim=0)
        count = change_mask_f.sum(dim=0)

        denom = torch.clamp(count, min=1.0)

        avg_diff = sum_diff / denom

        zero = torch.zeros_like(sum_diff)
        non_overlap_mask = (count == 1)
        overlap_mask = (count >= 2)

        rescales_tensor = torch.tensor(
            rescales,
            dtype=diffs.dtype,
            device=diffs.device,
        ).view((n,) + (1,) * (diffs.dim() - 1))

        weighted_sum = (diffs * change_mask_f * rescales_tensor).sum(dim=0)

        diff_final = zero
        diff_final = torch.where(overlap_mask, avg_diff, diff_final)
        diff_final = torch.where(non_overlap_mask, weighted_sum, diff_final)

        merged_sd[name] = base_sd[name] + diff_final

    for k in base_sd.keys():
        if k not in merged_sd:
            merged_sd[k] = base_sd[k].clone()

    return merged_sd

def save_merged_model_and_tokenizer(base_path, merged_sd, output_dir="./merged_model"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Save] Loading base model structure from {base_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float32,
        cache_dir="./cache",
    )
    print("[Save] Loading merged state_dict into model ...")
    model.load_state_dict(merged_sd)
    print(f"[Save] Saving merged model to {output_dir} ...")
    model.save_pretrained(output_dir)
    print(f"[Save] Loading tokenizer from base model: {base_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, cache_dir="./cache")
    print(f"[Save] Saving tokenizer to {output_dir} ...")
    tokenizer.save_pretrained(output_dir)
    print(f"[Save] Done. Merged model + tokenizer saved in: {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Agentic Reinforcement Merge CLI")
    
    # 默认值
    default_base_model = "Qwen/Qwen2.5-7B-Instruct"
    default_task_models = [
        "Gen-Verse/ReasonFlux-Coder-7B",
        "emrecanacikgoz/Qwen2.5-7B-Instruct-ToolRL-grpo-cold",
        "BytedTsinghua-SIA/RL-MemoryAgent-7B"
    ]

    parser.add_argument("--base_model", type=str, default=default_base_model, 
                        help=f"Path to the base model (default: {default_base_model})")
    
    parser.add_argument("--task_models", type=str, nargs='+', default=default_task_models, 
                        help=f"List of paths to task models")
    
    parser.add_argument("--output_dir", type=str, default="./MergedModel", help="Directory to save the merged model")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Threshold for parameter change detection")
    
    parser.add_argument("--merge_mode", type=str, default="arm-r-v2", 
                        choices=["arm", "arm-r", "arm-r-v2", "arm-ties", "arm-unique"],
                        help="Merge strategy to use")
    
    parser.add_argument("--rescale_factor", type=float, default=1.05, 
                        help="Rescale factor 'r' for arm-r and arm-r-v2 modes")
    
    parser.add_argument("--target_index", type=int, default=1, 
                        help="Index of target model for arm-unique mode")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    base_path = args.base_model
    task_paths = args.task_models
    threshold = args.threshold
    merge_mode = args.merge_mode

    print("Loading models...")
    print(f"Base: {base_path}")
    print(f"Tasks: {task_paths}")

    base_sd = load_state_dict(base_path)
    task_sds = [load_state_dict(p) for p in task_paths]

    stats = compute_overlap_stats_elementwise_n(base_sd, task_sds, threshold=threshold)
    n = len(task_paths)

    for i, (path, st) in enumerate(zip(task_paths, stats), start=1):
        total_changed = st["total_changed_elems"]
        total_elems = st["total_elems"]
        changed_ratio = 0.0 if total_elems == 0 else total_changed / total_elems * 100.0

        print(f"\n=== Task Model {i}: {path} ===")
        print(f"Total changed ELEMENTS: {total_changed}")
        print(f"Changed ratio vs base (by elements): {changed_ratio:.6f}%")

        for k in range(n):
            count_k = st["overlap_counts"][k]
            pct_k = st["overlap_pcts"][k]
            if k == 0:
                desc = "overlap with 0 other models (unique)"
            elif k == n - 1:
                desc = f"overlap with {k} other models (ALL {n} models)"
            else:
                desc = f"overlap with {k} other models"
            print(f"  {desc}: {count_k} ({pct_k:.4f}%)")

    task_vecs = compute_task_vectors(base_sd, task_sds)

    print(f"\n[Timer] Start merging with mode: {merge_mode}...")
    start_time = time.time()

    if merge_mode == "arm":
        print("\n[Merging] Using arm...")
        merged_sd = agentic_reinforcement_merge(base_sd, task_vecs, threshold=threshold)
    elif merge_mode == "arm-r":
        print(f"\n[Merging] Using arm-r with r={args.rescale_factor}...")
        merged_sd = agentic_reinforcement_merge_rescale(base_sd, task_vecs, threshold=threshold, r=args.rescale_factor)
    elif merge_mode == "arm-r-v2":
        print(f"\n[Merging] Using arm-r-v2 with r={args.rescale_factor}...")
        merged_sd = agentic_reinforcement_merge_rescale_v2(base_sd, task_vecs, threshold=threshold, r=args.rescale_factor)    
    elif merge_mode == "arm-ties":
        print("\n[Merging] Using arm-ties...")
        merged_sd = agentic_reinforcement_merge_ties(base_sd, task_vecs, threshold=threshold)   
    elif merge_mode == "arm-unique":
        print(f"\n[Merging] Using arm-unique...")
        print(f"Target Model Index: {args.target_index} ({task_paths[args.target_index]})")
        merged_sd = agentic_reinforcement_merge_extract_unique(base_sd, task_vecs, args.target_index, threshold=threshold)
    else:
        raise ValueError(f"Unknown merge_mode: {merge_mode}")

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    print(f"\n[Timer] Merge finished in {elapsed_seconds:.2f} seconds ({elapsed_seconds/60:.2f} minutes).")

    final_output_dir = os.path.join(args.output_dir, merge_mode)
    save_merged_model_and_tokenizer(base_path, merged_sd, output_dir=final_output_dir)