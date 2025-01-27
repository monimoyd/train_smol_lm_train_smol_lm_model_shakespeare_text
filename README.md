# train_smol_lm_train_smol_lm_model_shakespeare_text

In  this repository I have trained SmolLM2-135 model. The following techniques are used:

* set_float32_matmul_precision
*  Torch.compile for optimization
* Autocast
* Training it for 5000 steps while predicting every 500 steps on what it utters. Now fully stop the model and save a checkpoint. Now load this checkpoint and train for 50 more steps

  ## Training Logs

  okenizer_config.json: 100%
 26.0/26.0 [00:00<00:00, 1.27kB/s]
config.json: 100%
 665/665 [00:00<00:00, 28.0kB/s]
vocab.json: 100%
 1.04M/1.04M [00:00<00:00, 4.31MB/s]
merges.txt: 100%
 456k/456k [00:00<00:00, 2.17MB/s]
tokenizer.json: 100%
 1.36M/1.36M [00:00<00:00, 5.83MB/s]

Training:  10%|█         | 500/5000 [01:07<05:06, 14.68it/s]

Step 500: Loss = 0.7291

Training:  10%|█         | 502/5000 [01:08<16:04,  4.66it/s]

Model utterance: , barren not
 arms give barren

Training:  20%|██        | 1002/5000 [01:44<04:52, 13.66it/s]

Step 1000: Loss = 0.4729
Model utterance: , barren not
 arms
 barren

Training:  30%|███       | 1502/5000 [02:21<04:16, 13.66it/s]

Step 1500: Loss = 0.2971
Model utterance: , partly must I arms tire partly

Training:  40%|████      | 2002/5000 [02:57<03:38, 13.73it/s]

Step 2000: Loss = 0.1934
Model utterance: , partly must I arms tire partly

Training:  50%|█████     | 2502/5000 [03:33<03:03, 13.58it/s]

Step 2500: Loss = 0.1538
Model utterance: , partly in I arms be partly

Training:  60%|██████    | 3002/5000 [04:10<02:26, 13.61it/s]

Step 3000: Loss = 0.1381
Model utterance: , partly not it be tire barren

Training:  70%|███████   | 3502/5000 [04:46<01:49, 13.67it/s]

Step 3500: Loss = 0.1283
Model utterance: , barren in it arms tire barren

Training:  80%|████████  | 4002/5000 [05:22<01:12, 13.69it/s]

Step 4000: Loss = 0.1039
Model utterance: , barren in it be tire barren

Training:  90%|█████████ | 4502/5000 [05:59<00:36, 13.66it/s]

Step 4500: Loss = 0.0938
Model utterance: , barren in it arms tire barren

Training: 100%|██████████| 5000/5000 [06:35<00:00, 13.68it/s]

Step 5000: Loss = 0.0958
Model utterance: , barren in it be be barren
Checkpoint saved to smollm_checkpoint.pth

<ipython-input-3-215c27e8c476>:121: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(checkpoint_path))

Checkpoint loaded. Resuming training...


Training:   0%|          | 0/50 [00:00<?, ?it/s]

Training:   4%|▍         | 2/50 [00:00<00:02, 18.14it/s]

Training:   8%|▊         | 4/50 [00:00<00:03, 15.10it/s]

Training:  12%|█▏        | 6/50 [00:00<00:03, 14.35it/s]

Training:  16%|█▌        | 8/50 [00:00<00:02, 14.01it/s]

Training:  20%|██        | 10/50 [00:00<00:02, 13.87it/s]

Training:  24%|██▍       | 12/50 [00:00<00:02, 13.79it/s]

Training:  28%|██▊       | 14/50 [00:00<00:02, 13.62it/s]

Training:  32%|███▏      | 16/50 [00:01<00:02, 13.61it/s]

Training:  36%|███▌      | 18/50 [00:01<00:02, 13.56it/s]

Training:  40%|████      | 20/50 [00:01<00:02, 13.57it/s]

Training:  44%|████▍     | 22/50 [00:01<00:02, 13.57it/s]

Training:  48%|████▊     | 24/50 [00:01<00:01, 13.59it/s]

Training:  52%|█████▏    | 26/50 [00:01<00:01, 13.52it/s]

Training:  56%|█████▌    | 28/50 [00:02<00:01, 13.51it/s]

Training:  60%|██████    | 30/50 [00:02<00:01, 13.52it/s]

Training:  64%|██████▍   | 32/50 [00:02<00:01, 13.48it/s]

Training:  68%|██████▊   | 34/50 [00:02<00:01, 13.53it/s]

Training:  72%|███████▏  | 36/50 [00:02<00:01, 13.54it/s]

Training:  76%|███████▌  | 38/50 [00:02<00:00, 13.43it/s]

Training:  80%|████████  | 40/50 [00:02<00:00, 13.53it/s]

Training:  84%|████████▍ | 42/50 [00:03<00:00, 13.53it/s]

Training:  88%|████████▊ | 44/50 [00:03<00:00, 13.51it/s]

Training:  92%|█████████▏| 46/50 [00:03<00:00, 13.52it/s]

Training:  96%|█████████▌| 48/50 [00:03<00:00, 13.45it/s]

Training: 100%|██████████| 50/50 [00:04<00:00, 10.11it/s]

Checkpoint saved to final_checkpoint.pth
Training complete.


  

  


