# Generative Adversarial Imitation Learninig
Implementation of Generative Adversarial Imitation Learning(GAIL) using tensorflow  


## dependencies
python3.5  
tensorflow1.5rc   
gym

## Useage

Train experts  
python3 run_ppo.py   

Run GAIL  
python3 run_gail.py  

Run supervised learning  
python3 run_behavior_clone.py 

Test trained policy  
python3 test_policy.py

Tensorboard  
tensorboard --logdir log
