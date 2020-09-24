## TRAIN the head
CUDA_VISIBLE_DEVICES=4 python dialogGPT_discr.py --save_model --dataset sentiment --cached --epochs 100 > results/discriminator_training/sentiment_new.txt
CUDA_VISIBLE_DEVICES=4 python dialogGPT_discr.py --save_model --dataset daily_dialogue_act --cached --epochs 100 > results/discriminator_training/daily_dialogue_act_new.txt
CUDA_VISIBLE_DEVICES=4 python dialogGPT_discr.py --save_model --dataset empathetic_dialogue --cached --epochs 100 > results/discriminator_training/empathetic_dialogue_new.txt# CUDA_VISIBLE_DEVICES=4 python dialogGPT_discr.py --save_model --dataset TC_AG_NEWS --cached --epochs 50 > results/discriminator_training/TC_AG_NEWS_new.txt

## TRAIN the scorer
CUDA_VISIBLE_DEVICES=0 python train_score.py --dataset AmazonReviewFull > results/discriminator_training/Amazon5.txt
CUDA_VISIBLE_DEVICES=3 python train_score.py --dataset TC_AG_NEWS > results/discriminator_training/BERT_TEST_AG_NEWS.txt

