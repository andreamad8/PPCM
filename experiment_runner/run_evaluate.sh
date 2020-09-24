# ## very negative
# # DialGPT + PPLM 
CUDA_VISIBLE_DEVICES=1 python main.py -D sentiment --label_class 3 --length 30 --num_samples 10 --evaluate --verbose --sample_starter 10

# ## very positive
# # DialGPT 
# # DialGPT + PPLM 
CUDA_VISIBLE_DEVICES=1 python main.py -D sentiment --label_class 2 --length 30 --num_samples 10 --evaluate --verbose --sample_starter 10

# ## Daily Dialogue ACT class=1 Question
# # DialGPT 
# # DialGPT + PPLM 
CUDA_VISIBLE_DEVICES=1 python main.py -D daily_dialogue_act --label_class 1 --length 30 --num_samples 10 --evaluate --verbose --sample_starter 10

# ## Text Classficiation
# # DialGPT 
# # DialGPT + PPLM 
CUDA_VISIBLE_DEVICES=1 python main.py -D AG_NEWS --label_class 0 --length 30 --num_samples 10 --evaluate --verbose --sample_starter 10
CUDA_VISIBLE_DEVICES=1 python main.py -D AG_NEWS --label_class 1 --length 30 --num_samples 10 --evaluate --verbose --sample_starter 10
CUDA_VISIBLE_DEVICES=1 python main.py -D AG_NEWS --label_class 2 --length 30 --num_samples 10 --evaluate --verbose --sample_starter 10
CUDA_VISIBLE_DEVICES=1 python main.py -D AG_NEWS --label_class 3 --length 30 --num_samples 10 --evaluate --verbose --sample_starter 10

