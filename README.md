# CONDITION-BASED-SKETCH-COLOURISATION

Condition-Based Sketch Colourisation 
MSc Project — University of Surrey
student:Balaji Periyadurai
Supervisor: Prof. Yi-Zhe Song 
A deep learning system that converts black-and-white sketches into coloured images, guided by sparse user-provided colour hints. 
 Overview 
This project implements a Pix2Pix-style conditional GAN for sketch colourisation. The user provides a sketch image and a sparse
colour hint map. The model produces a full-colour image that follows the hints and respects the sketch structure. 
Architecture: - Generator: U-Net encoder-decoder with skip connections (7-channel input → 3-channel output) - Discriminator: 70×70
PatchGAN - Loss: Adversarial (BCE) + L1 reconstruction (λ=100) - Conditioning: sparse colour hint map (5% pixel coverage by default) 
