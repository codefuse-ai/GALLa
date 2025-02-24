import torch


# print out arguments in a nice way
def print_args(args, accelerator):
    # 计算所有键的最大字符串长度
    max_key_length = max(len(str(key)) for key in vars(args).keys())
    
    message = ""
    message += "====" * 40 + "\n"
    message += '\n'.join([f'{k:<{max_key_length}} : {v}' for k, v in vars(args).items()]) + "\n"
    message += "====" * 40 + "\n"
    accelerator.print(message)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def print_with_rank(accelerator, msg):
    print(accelerator.process_index, msg)


def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)


def print_rank_0_highlight(*message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print('=='*100)
            print(*message, flush=True)
            print('=='*100)
    else:
        print('=='*100)
        print(*message, flush=True)
        print('=='*100)


def print_highlight(*message):
    print('=='*100)
    print(*message)
    print('=='*100)


def get_computation_speed(batch_size_per_device, seq_len, step_time):

    return batch_size_per_device * seq_len / (step_time + 1e-12)


def touch_print(accelerator, batch, num_tokens=10):
    """touch first and last tokens and labels for debugging usage"""
    accelerator.print(f"step 1 batch shape: {batch['input_ids'].shape},\n"
                      f"last {num_tokens} labels: {batch['labels'][:, -num_tokens:]}"
                      f"last {num_tokens} loss mask: {batch['loss_mask'][:, -num_tokens:]}")
    accelerator.print(f"first {num_tokens} input_ids and loss_mask")
    for pt in range(1):
        accelerator.print(f"{batch['input_ids'][:, num_tokens * pt: num_tokens * pt + num_tokens]}")
        accelerator.print(f"{batch['loss_mask'][:, num_tokens * pt: num_tokens * pt + num_tokens]}")