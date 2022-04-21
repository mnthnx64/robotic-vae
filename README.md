# Robotic Hand Gesture Generator Using Variational AutoEncoders
**TL;DR**: A robot hand that internal representations of digits. A human can make gestures and see what that means to the robot. This is achieved by using Variational AutoEncoders.

## Setup
```properties
conda env create -f environment.yml
```

## Model Description
The model can be seperated into two parts:
<details>
  <summary>Variational AutoEncoder</summary>

  ------------------------------------------------
  ### Overview
  - The VAE will be used in the inital training process. 
  - The VAE will learn a latent space that contains `20 elements` that represent the position of the robotic hand.
  - During testing, the decoder will be used to generate digits for new lataent space members.
  ### Layers
  ----------- WIP --------------
</details>

<details>
  <summary>Classifier</summary>

  ------------------------------------------------
  ### Overview
  - It classifies the latent space into digits.
  - This can be used as a cross checker in case the decoder of the VAE fails to generate a valid output.
  ### Layers
  ----------- WIP --------------
</details>

----------------
## Estimates:
| Task              | Time required | Assignee | Current Status |
|-------------------|---------------|----------|----------------|
| Model VAE         | 1 day         | Manthan  | in progress    |
| Model Classifier  | 1 day         | Sowmith  | in progress    |
| Train             | 3 days        |          |                |
| Validate          | 1 day         |          |                |
| Hand Recognition  | 3 days        |          |                |

-----------
## Questions to Answer with Reason:
- Justify the layers of the model
- How to train the model?
  - Train the VAE first and then the classifer or
  - Train both VAE and classifer simultaneously?

## Training Pipeline
Train the VAE first
Use the VAE's encoder and the classifer to train the classifer
