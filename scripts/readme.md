#  Scripts

##  Getting Started
You can get started in either of the following ways:

### 1. Running the script directly
1. Clone the repository
2. Run the script using the following command:
```bash
python3 mh_vae.py mnist
python3 mh_sae.py mnsit
python3 mh_vae_train.py mnist
python3 mh_sae_train.py mnist
```

### 2. Using the Docker image
1. Run the following command to build the Docker image and run the container:
```bash
./docker.sh
```
2. Inside the container you can run the scripts using the following commands:
```bash
python3 mh_vae.py mnist
python3 mh_sae.py mnsit
python3 mh_vae_train.py mnist
python3 mh_sae_train.py mnist
```
In both cases, if you observe a progress bar, it indicates that the training has begun. Once the training is completed, the model weights will automatically be saved in memory for future use.


##  Files
The followings are the scripts for the different translators used in the paper. 

- `mh_vae.py`: Script for the VAE architecture and a simple training showcase.
- `mh_sae.py`: Script for the SAE architecture and a simple training showcase.
- `mh_style_transform.py`: Script for the style transfer architecture and a simple style transfer showcase.
- `mh_cyclegan.py`: Script for the CycleGAN architecture and a simple CycleGAN showcase.

The followings are the scripts for their training and evaluation.

- `mh_vae_train.py`: Script for training the VAE architecture.
- `mh_sae_train.py`: Script for training the SAE architecture.
- `mh_sae_eval.py`: Script for evaluating the SAE architecture.
- `mh_vae_eval.py`: Script for evaluating the VAE architecture.
- `mh_offline_eval.py`: Script for evaluating the offline testing of ADS-DNNs.
- `mh_sae_vae_train.py` Script for training the SAEVAE architecture.
- `mh_style_transform_eval.py`: Script for evaluating the style transfer architecture.
- `mh_ds.py`: Script for loading the datasets. Note that this script needs all the datasets to be downloaded and put in the same directory as the script.
