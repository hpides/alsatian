import time

import torch
from transformers import BertForSequenceClassification

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                          num_labels=2)  # 2 for binary classification

    batch_size = 256
    dummy_input_id = torch.randint(0, 100, (batch_size, 128))  # Replace 256 with your desired batch size
    dummy_attention_mask = torch.randint(0, 2, (batch_size, 128))  # Binary mask
    dummy_label = torch.randint(0, 2, (batch_size,))
    num_batches = 20

    model.eval()
    model.to(device)
    input_ids, attention_mask, labels = dummy_input_id, dummy_attention_mask, dummy_label
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    end_to_end_start = time.time()
    batch_times = []
    with torch.no_grad():
        for i in range(num_batches):
            # start = time.time()
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            outputs = model(input_ids, attention_mask=attention_mask)
            ender.record()
            torch.cuda.synchronize()  # WAIT FOR GPU SYNC
            elapsed = starter.elapsed_time(ender)
            batch_times.append(elapsed * 10 ** -3)
            # time.sleep(2)

    end_to_end_end = time.time()
    print(f'end to end: {end_to_end_end - end_to_end_start}')
    print(f'sum: {sum(batch_times)}')
    print(batch_size * num_batches / sum(batch_times) * 2.35)
    print(batch_times)
