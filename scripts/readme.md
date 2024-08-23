#  Scripts

##  Getting Started
You can get started in either of the following ways:

### 1. Running the script directly
1. Clone the repository
2. Run the script using the following command:
```bash
python3 mh_vae.py
```

### 2. Using the Docker image
1. Run the following command to build the Docker image and run the container:
```bash
./docker.sh
```

##  Files
The followings are the scripts for the different translators used in the paper. 

- `mh_vae.py`: Script for the VAE architecture and a simple training showcase.
- `mh_sae.py`: Script for the SAE architecture and a simple training showcase.
- `mh_style_transform.py`: Script for the style transfer architecture and a simple style transfer showcase.
- `mh_cyclegan.py`: Script for the CycleGAN architecture and a simple CycleGAN showcase.

The followings are the scripts for their training and evaluation. Note that in each script, there is a block of code with the comment `##  loading data...` that loads the data. You need to replace this block with your own data.

- `mh_vae_train.py`: Script for training the VAE architecture.
- `mh_sae_train.py`: Script for training the SAE architecture.
- `mh_sae_vae_train.py` Script for training the SAE and VAE architecture.
- `mh_style_transform_train.py`: Script for training the style transfer architecture.
