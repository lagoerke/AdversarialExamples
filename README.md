# AdversarialExamples
Code of Bachelor's Thesis "Adversarial Examples - Analyzing Attacks and Defenses"

The files "mnist_splitted.py" and "cifar_splitted.py" conatin code for creating the original and the distilled model for according dataset. This code was executed on https://vispa.physik.rwth-aachen.de/ for better performance. 
Afterwards the resulting models were downloaded to a local drive.

The files "loopMNIST.py" and "loopCIFAR.py" execute the code of "MNISTFGSM.py" and "CIFARFGSM.py" respecively. 
The loop files define the folders in which the already downloaded models are and into which the adversarial images should be saved.

"MNISTFGSM.py" and "CIFARFGSM.py" create the adversarial images.
